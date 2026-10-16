"""Cloud-only resumable HTTP range download; bounded workers, immutable ETag."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import urllib.request


def parallel_download(url, target, workers=8):
    if not Path('/etc/dsw').exists():
        raise RuntimeError('Cloud-only downloader')
    target = Path(target)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, headers={'Range': 'bytes=0-0', 'User-Agent': 'ai-project-academic-collector/1.0'})
    with opener.open(req, timeout=30) as response:
        match = re.fullmatch(r'bytes 0-0/(\d+)', response.headers.get('Content-Range', ''))
        if response.status != 206 or not match:
            raise ValueError('Source does not support exact range requests')
        total = int(match[1])
        etag = response.headers.get('ETag')
        source_url = response.url  # Signed redirect stays in memory only.
        response.read(1)
    if not etag or etag.startswith('W/'):
        raise ValueError('A strong source ETag is required for parallel download')
    part = target.with_name(target.name + '.part')
    journal = target.with_name(target.name + '.ranges.json')
    if journal.exists():
        state = json.loads(journal.read_text())
        if state['etag'] != etag or state['total'] != total:
            raise ValueError('Source changed; refuse to mix snapshots')
    else:
        # Preserve the sequential prefix already downloaded. ZIP CRC is checked
        # by the corpus reader before any training-ready manifest is emitted.
        prefix = min(part.stat().st_size, total) if part.exists() else 0
        step = 32 * 1024 ** 2
        state = {'etag': etag, 'total': total, 'prefix_bytes': prefix,
                 'ranges': [[begin, min(begin + step, total) - 1] for begin in range(prefix, total, step)], 'completed': []}
        journal.write_text(json.dumps(state))
    complete_ranges = len(set(state['completed'])) == len(state['ranges'])
    if target.exists():
        # A crash may happen after rename but before the caller writes its receipt.
        # Never recreate an empty .part and trust the old completed-range journal.
        if not complete_ranges or target.stat().st_size != total:
            raise ValueError('Existing final archive conflicts with the range journal')
        with target.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        return {'file': str(target), 'bytes': total, 'sha256': digest, 'url': url,
                'source_etag': etag, 'workers': workers, 'proxy': False}
    if not part.exists() and (state['completed'] or state['prefix_bytes']):
        raise ValueError('Partial file is missing; refuse to trust its completion journal')
    mutex = threading.Lock()
    fd = os.open(part, os.O_CREAT | os.O_RDWR, 0o600)
    def transfer(index):
        begin, end = state['ranges'][index]
        headers = {'Range': f'bytes={begin}-{end}', 'If-Match': etag}
        for attempt in range(3):
            try:
                request = urllib.request.Request(source_url, headers=headers)
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=30) as response:
                    expected = f'bytes {begin}-{end}/{total}'
                    if response.status != 206 or response.headers.get('Content-Range') != expected:
                        raise ValueError('Range mismatch')
                    offset = begin
                    while block := response.read(1024 ** 2):
                        if offset + len(block) > end + 1:
                            raise ValueError('Range exceeded expected size')
                        written = 0
                        while written < len(block):
                            written += os.pwrite(fd, block[written:], offset + written)
                        offset += len(block)
                    if offset != end + 1:
                        raise ValueError('Truncated range')
                with mutex:
                    os.fsync(fd)
                    state['completed'].append(index)
                    temporary = journal.with_suffix('.json.tmp')
                    temporary.write_text(json.dumps(state))
                    temporary.replace(journal)
                    print('RANGE_READY', len(state['completed']), '/', len(state['ranges']), flush=True)
                return
            except Exception:
                if attempt == 2:
                    raise
    try:
        pending = [i for i in range(len(state['ranges'])) if i not in state['completed']]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(transfer, pending))
        os.ftruncate(fd, total)
        os.fsync(fd)
    finally:
        os.close(fd)
    with part.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    part.replace(target)
    return {'file': str(target), 'bytes': total, 'sha256': digest, 'url': url,
            'source_etag': etag, 'workers': workers, 'proxy': False}
