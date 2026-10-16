"""Shared cloud download helper: verified size/hash, direct connection and retries."""
import hashlib
from pathlib import Path
import time
import urllib.parse
import urllib.request


def download(url, target, expected_size=None, expected_sha=None):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (expected_size is None or target.stat().st_size == expected_size):
        digest = hashlib.file_digest(target.open('rb'), 'sha256').hexdigest()
        if expected_sha and digest == expected_sha:
            return {'file': str(target), 'bytes': target.stat().st_size, 'sha256': digest, 'url': url}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    temporary = target.with_name(target.name + '.part')
    for attempt in range(4):
        try:
            digest = hashlib.sha256()
            count = 0
            request = urllib.request.Request(url, headers={'User-Agent': 'ai-project-academic-collector/1.0'})
            with opener.open(request, timeout=90) as response, temporary.open('wb') as out:
                final_url = response.url
                while block := response.read(4 * 1024 * 1024):
                    out.write(block)
                    digest.update(block)
                    count += len(block)
            if expected_size is not None and count != expected_size:
                raise ValueError('Unexpected file size: ' + target.name)
            if expected_sha and digest.hexdigest() != expected_sha:
                raise ValueError('SHA256 mismatch: ' + target.name)
            temporary.replace(target)
            print('Downloaded', target.name, count, flush=True)
            # Redirect URLs may be signed; do not store their query strings.
            return {'file': str(target), 'bytes': count, 'sha256': digest.hexdigest(),
                    'url': url, 'download_host': urllib.parse.urlsplit(final_url).hostname}
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))

