"""Cloud-only ModelScope snapshot download, pinned per-file commits and hashes."""
import argparse
import concurrent.futures
import fcntl
import json
import shutil
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from cloud_download import download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--root', type=Path, default=Path('/mnt/workspace/qwen35_update'))
    args = parser.parse_args()
    if not Path('/etc/dsw').exists() or not str(args.root.resolve()).startswith('/mnt/workspace/'):
        raise SystemExit('Run this downloader on Alibaba DSW under /mnt/workspace only.')
    if args.model not in {'Qwen/Qwen3.5-4B', 'Qwen/Qwen3.5-4B-Base', 'Qwen/Qwen3-4B-Base'}:
        raise SystemExit('Model is not in the approved 4B experiment set.')
    destination = args.root / 'base' / args.model.split('/')[-1]
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / '.download.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = destination / 'download-manifest.json'
        if manifest.exists():
            snapshot = json.loads(manifest.read_text())
        else:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            url = f'https://modelscope.cn/api/v1/models/{args.model}/repo/files?Revision=master&Recursive=true'
            with opener.open(url, timeout=30) as response:
                listing = json.load(response)
            files = [f for f in listing['Data']['Files'] if f['Type'] != 'tree']
            for item in files:
                path = Path(item['Path'])
                if path.is_absolute() or '..' in path.parts or not item.get('Sha256') or not item.get('Revision'):
                    raise ValueError('Unsafe path or missing immutable file metadata')
            snapshot = {'model': args.model, 'captured_at': datetime.now(timezone.utc).isoformat(),
                        'files': files, 'location': 'Alibaba DSW', 'proxy': False}
            manifest.write_text(json.dumps(snapshot, indent=2))
        needed = sum(f['Size'] for f in snapshot['files'] if not (destination / f['Path']).exists())
        if shutil.disk_usage(destination).free < needed + 2 * 1024 ** 3:
            raise RuntimeError('Insufficient disk space for snapshot plus 2 GiB reserve')
        def fetch(item):
            # Each file uses the immutable commit at which its captured hash was recorded.
            query = urllib.parse.urlencode({'Revision': item['Revision'], 'FilePath': item['Path']})
            return download(f'https://modelscope.cn/api/v1/models/{args.model}/repo?{query}',
                            destination / item['Path'], item['Size'], item['Sha256'])
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            receipts = list(pool.map(fetch, snapshot['files']))
        receipt = {'model': args.model, 'completed_at': datetime.now(timezone.utc).isoformat(),
                   'files': receipts, 'proxy': False, 'download_location': 'Alibaba DSW'}
        temp = destination / 'download-complete.json.tmp'
        temp.write_text(json.dumps(receipt, indent=2))
        temp.replace(destination / 'download-complete.json')
        print('MODEL_READY', destination, flush=True)


if __name__ == '__main__':
    main()
