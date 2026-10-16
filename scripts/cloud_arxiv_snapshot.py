"""Download arXiv's Kaggle metadata snapshot on DSW and select new submissions."""
import argparse
from collections import Counter
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
import fcntl
import hashlib
import json
from pathlib import Path
import zipfile

from cloud_download import download
from cloud_parallel_download import parallel_download

SOURCE = 'https://www.kaggle.com/api/v1/datasets/download/Cornell-University/arxiv'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/mnt/workspace/qwen35_update/data/arxiv_snapshot'))
    p.add_argument('--since', default='2026-03-03')
    p.add_argument('--until', default=date.today().isoformat())
    args = p.parse_args()
    if not Path('/etc/dsw').exists():
        raise SystemExit('Cloud-only download: run on DSW.')
    start, end = date.fromisoformat(args.since), date.fromisoformat(args.until)
    if start > end:
        raise ValueError('Invalid date range')
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        archive = args.root / 'arxiv-metadata.zip'
        receipt_path = args.root / 'download_receipt.json'
        old = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        receipt = (download(SOURCE, archive, old.get('bytes'), old.get('sha256'))
                   if archive.exists() and old else parallel_download(SOURCE, archive))
        receipt['retrieved_at'] = old.get('retrieved_at', datetime.now(timezone.utc).isoformat())
        receipt_temp = receipt_path.with_suffix('.json.tmp')
        receipt_temp.write_text(json.dumps(receipt, indent=2))
        receipt_temp.replace(receipt_path)
        counts, chars, earliest, latest = Counter(), 0, None, None
        seen = set()
        outputs = {split: (args.root / f'{split}.jsonl.part').open('w', encoding='utf-8') for split in ('train', 'validation')}
        try:
            with zipfile.ZipFile(archive) as zipped:
                with zipped.open('arxiv-metadata-oai-snapshot.json') as source:
                    for line in source:
                        row = json.loads(line)
                        counts['snapshot_records'] += 1
                        versions = row.get('versions') or []
                        if not versions:
                            continue
                        first = min(parsedate_to_datetime(v['created']).date() for v in versions)
                        last = max(parsedate_to_datetime(v['created']).date() for v in versions)
                        if not start <= first <= end or last > end:
                            continue
                        categories = row.get('categories', '').split()
                        if not any(c.startswith(('cs.CL', 'cs.LG', 'math.', 'quant-ph', 'physics.', 'stat.')) for c in categories):
                            continue
                        title = ' '.join(row.get('title', '').split())
                        abstract = ' '.join(row.get('abstract', '').split())
                        if len(abstract) < 120 or 'withdrawn' in (title + ' ' + (row.get('comments') or '')).lower():
                            continue
                        digest = hashlib.sha256((title + '\n' + abstract).encode()).hexdigest()
                        if digest in seen:
                            continue
                        seen.add(digest)
                        paper_id = row['id']
                        split = 'validation' if int(hashlib.sha256(paper_id.encode()).hexdigest()[:8], 16) % 20 == 0 else 'train'
                        record = {'paper_id': paper_id, 'title': title,
                            'text': f'{title}\nPublished: {first.isoformat()}\nResearch abstract (claims by the authors):\n{abstract}',
                            'source_url': 'https://arxiv.org/abs/' + paper_id, 'published': first.isoformat(),
                            'updated': last.isoformat(), 'categories': categories, 'metadata_license': 'CC0-1.0',
                            'paper_license': row.get('license'), 'journal_ref': row.get('journal-ref'),
                            'verification': 'author-provided abstract; not independently verified',
                            'sha256': digest, 'split': split, 'retrieved_at': receipt['retrieved_at']}
                        outputs[split].write(json.dumps(record, ensure_ascii=False) + '\n')
                        counts[split] += 1
                        chars += len(record['text'])
                        earliest = min(earliest or first, first)
                        latest = max(latest or first, first)
            if counts['train'] < 500 or counts['validation'] < 20:
                raise RuntimeError('Snapshot lacks sufficient post-release records; do not start training.')
        finally:
            for out in outputs.values():
                out.close()
        files = {}
        for split in outputs:
            path = args.root / f'{split}.jsonl'
            path.with_suffix('.jsonl.part').replace(path)
            with path.open('rb') as source:
                files[split] = {'sha256': hashlib.file_digest(source, 'sha256').hexdigest(), 'bytes': path.stat().st_size}
        manifest = {'status': 'completed', 'since_inclusive': args.since, 'until_inclusive': args.until,
            'earliest_selected': str(earliest), 'latest_selected': str(latest), 'counts': dict(counts),
            'characters': chars, 'files': files, 'snapshot_sha256': receipt['sha256'],
            'source': 'https://www.kaggle.com/datasets/Cornell-University/arxiv',
            'license_source': 'https://info.arxiv.org/help/api/tou.html',
            'qwen35_4b_release': '2026-03-02', 'knowledge_cutoff': 'not publicly established',
            'download_location': 'Alibaba DSW', 'proxy': False,
            'scope': 'Scientific abstract metadata only; original publication date is filtered, not dataset update date.'}
        temporary = args.root / 'manifest.json.tmp'
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(args.root / 'manifest.json')
        print('CORPUS_READY', json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
