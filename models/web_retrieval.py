"""Bounded local web search + page excerpts, without a model API.

No model API is involved: web results become cited reference material for the
local model. Search HTML may change; failures are reported, never invented.
"""
import argparse
import base64
import concurrent.futures
import http.client
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import ssl
import sys
import subprocess
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote, urlencode, urljoin, urlsplit

FETCH_DIAGNOSTICS = []


def time_left(deadline, stage):
    budget = deadline - time.monotonic()
    if budget <= 0:
        exc = TimeoutError('Retrieval deadline exceeded')
        exc.stage = stage
        raise exc
    return budget

if __package__:
    from .retrieval_policy import (retrieval_meta_question, company_fact_question,
                                   relevant_text, relevance_terms)
else:
    from retrieval_policy import (retrieval_meta_question, company_fact_question,
                                  relevant_text, relevance_terms)


def clean(text):
    return re.sub(r'[\x00-\x08\x0b-\x1f\x7f]', '', text).strip()


def network_forbidden(question):
    # Negations and hypothetical lack of network take precedence over keywords.
    # This also takes precedence over /web on in the chat CLI.
    return bool(re.search(r'(?:不要|不准|禁止|无需|不必|不用|不能|无法|不允许|不需要|不想|别|不)(?:进行|使用|执行)?(?:联网|上网|搜索|检索)|'
                 r'(?:离线|断网)(?:回答|推理|模式)?|(?:没有|无)(?:网络|联网能力)|'
                 r'\b(?:without\s+(?:internet|web)|offline|(?:do\s+not|don[’\']t|cannot|can[’\']t)\s+(?:browse|search))\b', question, re.I))


def should_search(question):
    """Auto-search heuristic, with explicit network prohibitions first."""
    if network_forbidden(question) or retrieval_meta_question(question):
        return False
    # Match a request such as "请搜索 X", not a mention such as "检索是什么意思".
    request = re.search(
        r'(?:^|[。！？；\n，,])\s*(?:(?:请你|请|帮我|帮忙|麻烦你|麻烦|能否|能不能|可以|你|先|再)\s*){0,3}'
        r'(?:联网(?:搜索|检索|查找)?|上网(?:搜索|查找)?|搜索|检索|查找|查询|搜一下|查一下|查查)\s*(.+)', question)
    explicit = request and not re.match(r'(?:是什么|是什么意思|有什么用|的(?:原理|作用|意思)|怎么|如何)', request[1])
    english_request = re.search(r'^(?:please\s+)?(?:search|look\s+up|browse)\s+(?!means?\b|meaning\b|is\b).+', question, re.I)
    timely = re.search(r'最新|今天|昨日|昨天|最近|今年|据说|20(?:2[5-9]|[3-9]\d)年?|\b(latest|today|current)\b', question, re.I)
    # “现在可以了吗” is a conversation follow-up, not a useful web query.
    current_topic = re.search(r'现在|目前', question) and re.search(
        r'进展|新闻|价格|排名|最强|哪[个家]|GPT|Qwen|Kimi|DeepSeek', question, re.I)
    return bool(explicit or english_request or timely or current_topic or company_fact_question(question))


def search_local(query, limit=3):
    """Bound the entire search, including DNS and page workers, to 35 seconds."""
    try:
        command = [sys.executable, '-m', 'models.web_retrieval', '--stdio']
        root = Path(__file__).resolve().parent.parent
        if os.name != 'nt' and Path('/proc/sys/kernel/osrelease').exists() and 'microsoft' in Path('/proc/sys/kernel/osrelease').read_text().lower():
            # Use Windows only as a network worker: bind physical sockets so
            # a WSL TUN route cannot silently spend the user's VPN allowance.
            candidates = sorted(Path('/mnt/c/Users').glob('*/AppData/Local/Programs/Python/Python*/python.exe'))
            python = os.environ.get('AI_WEB_PYTHON') or (str(candidates[-1]) if candidates else None)
            if not python:
                raise ConnectionError('Windows Python is required for physical-interface retrieval')
            winroot = subprocess.run(['wslpath', '-w', str(root)], capture_output=True, text=True, check=True, timeout=3).stdout.strip()
            command = [python, '-c', "import sys,runpy;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('models.web_retrieval',run_name='__main__')", winroot, '--stdio']
        result = subprocess.run(
            command,
            input=json.dumps({'query': query[:300], 'limit': limit}),
            text=True, capture_output=True, timeout=35, check=True,
            cwd=str(root))
        payload = json.loads(result.stdout)
        if (not isinstance(payload, dict)
                or payload.get('status') not in ('ok', 'limited', 'empty', 'failed')
                or not isinstance(payload.get('hits'), list)
                or not isinstance(payload.get('notice', ''), str)):
            raise ValueError('Invalid search worker response')
        for hit in payload['hits']:
            if (not isinstance(hit, dict) or hit.get('kind') != 'web'
                    or not all(isinstance(hit.get(key), str) and hit[key].strip()
                               for key in ('id', 'text', 'url', 'source', 'retrieved_at'))
                    or urlsplit(hit['url']).scheme not in ('http', 'https')
                    or not urlsplit(hit['url']).hostname
                    or hit.get('fetch_status') not in ('page_excerpt', 'search_snippet_only')
                    or not relevant_text(query, hit['text'])):
                raise ValueError('Invalid search worker evidence')
        if ((payload['status'] in ('ok', 'limited')) != bool(payload['hits'])):
            raise ValueError('Inconsistent search worker status')
        has_page = any(hit['fetch_status'] == 'page_excerpt' for hit in payload['hits'])
        if payload['hits'] and ((payload['status'] == 'ok') != has_page):
            raise ValueError('Inconsistent search worker evidence quality')
        return payload
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {'hits': [], 'status': 'failed',
                'error': 'timeout' if isinstance(exc, subprocess.TimeoutExpired) else type(exc).__name__,
                'failure_stage': 'worker_deadline' if isinstance(exc, subprocess.TimeoutExpired) else 'worker_response',
                'notice': f'本机联网检索失败（{type(exc).__name__}），本轮没有取得联网证据。'}


def public_target(url, *, qwen_card=False, timeout=6):
    parts = urlsplit(url)
    if parts.scheme not in ('https', 'http') or not parts.hostname or parts.username or parts.password:
        raise ValueError('Only public HTTP(S) URLs without credentials are allowed')
    if parts.port not in (None, 80, 443):
        raise ValueError('Nonstandard destination port')
    host = parts.hostname.encode('idna').decode('ascii')
    port = parts.port or (443 if parts.scheme == 'https' else 80)
    if os.name == 'nt':
        from .direct_network import resolve
        addresses = resolve(host, port, timeout=timeout)
    else:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    internal_modelscope = (qwen_card and parts.scheme == 'https' and host == 'modelscope.cn'
        and re.fullmatch(r'/api/v1/models/Qwen/Qwen[\w.\-]+/repo', parts.path)
        and parse_qs(parts.query) == {'Revision': ['master'], 'FilePath': ['README.md']}
        and all(ipaddress.ip_address(row[4][0]).is_global or ipaddress.ip_address(row[4][0]) in ipaddress.ip_network('10.224.0.0/16')
                for row in addresses))
    if not addresses or (not internal_modelscope and any(not ipaddress.ip_address(row[4][0]).is_global for row in addresses)):
        raise ValueError('Private, loopback, reserved or mixed DNS destination')
    # Prefer IPv4 on DSW, where IPv6 DNS records often exist without a route.
    addresses.sort(key=lambda row: row[0] != socket.AF_INET)
    return parts, host, addresses


def fetch(url, *, timeout=7, max_bytes=1_000_000, qwen_card=False, _deadline=None):
    started = time.monotonic()
    stage = 'search_html' if urlsplit(url).hostname in ('cn.bing.com', 'www.bing.com') else 'page_excerpt'
    record = {'stage': stage, 'status': 'ok'}
    try:
        return _fetch(url, timeout=timeout, max_bytes=max_bytes, qwen_card=qwen_card, _deadline=_deadline)
    except Exception as exc:
        record.update(status='failed', error=type(exc).__name__, failure_stage=getattr(exc, 'stage', stage))
        if not hasattr(exc, 'stage'):
            exc.stage = stage
        raise
    finally:
        record['seconds'] = round(time.monotonic() - started, 3)
        FETCH_DIAGNOSTICS.append(record)


def _fetch(url, *, timeout=7, max_bytes=1_000_000, qwen_card=False, _deadline=None):
    """Pin the validated public address, retaining TLS hostname verification.

    No environment proxy; revalidate every redirect (including DNS rebinding).
    The outer cloud subprocess also has an end-to-end deadline.
    """
    deadline = time.monotonic() + timeout
    if _deadline is not None:
        deadline = min(deadline, _deadline)
    for _ in range(4):
        parts, host, addresses = public_target(url, qwen_card=qwen_card, timeout=time_left(deadline, 'dns'))
        port = parts.port or (443 if parts.scheme == 'https' else 80)
        remaining = time_left(deadline, 'connect')
        if os.name == 'nt':
            from .direct_network import connect
            raw = connect(addresses[0][4][0], port, remaining)
        else:
            raw = socket.create_connection(addresses[0][4][:2], timeout=remaining)
        connection = http.client.HTTPConnection(host, port, timeout=remaining)
        stage = 'tls'
        try:
            raw.settimeout(time_left(deadline, stage))
            transport = ssl.create_default_context().wrap_socket(raw, server_hostname=host) if parts.scheme == 'https' else raw
            connection.sock = transport
            stage = 'response_headers'
            transport.settimeout(time_left(deadline, stage))
            request_path = (parts.path or '/') + ('?' + parts.query if parts.query else '')
            connection.request('GET', request_path, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; ai-project-retrieval/1.0)',
                'Accept': 'text/html,application/xhtml+xml,text/plain;q=0.8', 'Accept-Encoding': 'identity',
            })
            transport.settimeout(time_left(deadline, stage))
            response = connection.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                url = urljoin(url, response.getheader('Location', ''))
                continue
            if response.status != 200:
                raise ValueError(f'HTTP {response.status}')
            kind = response.getheader('Content-Type', '').lower()
            if not any(t in kind for t in ('text/html', 'application/xhtml+xml', 'text/plain', 'text/markdown')) and not qwen_card:
                raise ValueError('Not a readable text page')
            body = bytearray()
            stage = 'response_body'
            while len(body) < max_bytes:
                transport.settimeout(time_left(deadline, stage))
                # read1 prevents waiting for a full large buffer from a slow host.
                block = response.read1(min(32768, max_bytes - len(body)))
                if not block:
                    break
                body.extend(block)
            charset = response.headers.get_content_charset() or 'utf-8'
            try:
                text = bytes(body).decode(charset, errors='replace')
            except LookupError:
                text = bytes(body).decode('utf-8', errors='replace')
            return clean(text), url
        except Exception as exc:
            if not hasattr(exc, 'stage'):
                exc.stage = stage
            raise
        finally:
            connection.close()
            raw.close()
    raise ValueError('Too many redirects')


class SearchHTML(HTMLParser):
    """Extract only organic Bing result cards, not ads or generated answers."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.items, self.item, self.depth, self.heading, self.paragraph = [], None, 0, False, False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'li':
            if self.item is not None:
                self.depth += 1
            elif 'b_algo' in attrs.get('class', '').split():
                self.item = {'title': '', 'url': '', 'snippet': ''}
                self.depth = 1
        if self.item is not None:
            if tag == 'h2':
                self.heading = True
            if tag == 'p':
                self.paragraph = True
            if tag == 'a' and self.heading and not self.item['url']:
                self.item['url'] = attrs.get('href', '')

    def handle_endtag(self, tag):
        if tag == 'h2':
            self.heading = False
        if tag == 'p':
            self.paragraph = False
        if tag == 'li' and self.item is not None:
            self.depth -= 1
            if self.depth == 0:
                if self.item['url'] and self.item['title']:
                    self.items.append(self.item)
                self.item = None

    def handle_data(self, data):
        if self.item is not None:
            if self.heading:
                self.item['title'] += data
            elif self.paragraph:
                self.item['snippet'] += data


class PageHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'nav', 'footer', 'header', 'noscript', 'svg'):
            self.skip += 1
        if not self.skip and tag in ('p', 'h1', 'h2', 'h3', 'li', 'br', 'div', 'section', 'tr'):
            self.parts.append('\n')
        if not self.skip and tag in ('td', 'th'):
            self.parts.append(' | ')

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'nav', 'footer', 'header', 'noscript', 'svg'):
            self.skip = max(0, self.skip - 1)
        if not self.skip and tag in ('p', 'li', 'div', 'section', 'tr'):
            self.parts.append('\n')

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

    def paragraphs(self, min_chars=40):
        return [clean(' '.join(p.split())) for p in ''.join(self.parts).split('\n') if len(p.strip()) > min_chars]


def unwrap_url(url):
    url = urljoin('https://www.bing.com', url)
    parts = urlsplit(url)
    if parts.hostname in ('www.bing.com', 'cn.bing.com') and parts.path.startswith('/ck/a'):
        value = parse_qs(parts.query).get('u', [''])[0]
        if value.startswith('a1'):
            value = value[2:]
            url = base64.urlsafe_b64decode(value + '=' * (-len(value) % 4)).decode('utf-8')
    return url


def search_web(query, limit=3, *, _retry=True, _original_query=None, _deadline=None):
    query = clean(query)[:300]
    if not query or not 1 <= limit <= 5:
        raise ValueError('Empty query or invalid result count')
    fetched = datetime.now(timezone.utc).isoformat(timespec='seconds')
    if _deadline is None:
        _deadline = time.monotonic() + 24
    evidence_query = _original_query or query
    # Preserve versioned model names: search engines otherwise rewrite 3.5 to 3.
    search_query = re.sub(r'(?<![\w\"])([A-Za-z][A-Za-z0-9_-]*\d+\.\d+(?:[-A-Za-z0-9.]*)?)(?![\w\"])', r'"\1"', query)
    html, _ = fetch('https://cn.bing.com/search?' + urlencode(
        {'q': search_query, 'count': 10, 'setlang': 'zh-hans'}, quote_via=quote), timeout=9, _deadline=_deadline)
    parser = SearchHTML()
    parser.feed(html)
    selected, seen = [], set()
    terms = relevance_terms(evidence_query)
    distinctive = {t for t in terms if re.search(r'[a-z]', t) and re.search(r'\d', t)}
    for item in parser.items:
        try:
            url = unwrap_url(item['url'])
            parts = urlsplit(url)
            if parts.scheme not in ('https', 'http') or not parts.hostname or url in seen:
                continue
            if parts.hostname.endswith('bing.com'):
                continue
            evidence = item['title'] + ' ' + item['snippet']
            # Discard obvious off-topic responses rather than supplying false evidence.
            if not relevant_text(evidence_query, evidence):
                continue
            seen.add(url)
            selected.append({**item, 'url': url})
        except (ValueError, UnicodeError):
            continue
    primary_hosts = {'qwen.ai', 'github.com', 'huggingface.co', 'arxiv.org', 'openai.com', 'deepseek.com', 'kimi.com'}
    selected.sort(key=lambda item: (urlsplit(item['url']).hostname or '').removeprefix('www.') not in primary_hosts)
    selected = selected[:limit]
    if not selected and distinctive and _retry:
        # Some regional searches rewrite long mixed Chinese/model-version queries.
        # A second, explicitly narrower query is recorded, not hidden as a hit.
        narrowed = ' '.join(sorted(distinctive, key=lambda term: -len(term)))
        retry = search_web(narrowed, limit, _retry=False, _original_query=evidence_query, _deadline=_deadline)
        retry['original_query'] = query
        return retry
    def read(item):
        status, excerpt, final_url = 'search_snippet_only', '', item['url']
        try:
            parts = urlsplit(item['url'])
            model = None
            if parts.hostname == 'huggingface.co':
                model = re.fullmatch(r'/Qwen/(Qwen[\w.\-]+)/?', parts.path)
            elif parts.hostname in ('www.modelscope.cn', 'modelscope.cn'):
                model = re.fullmatch(r'/models/Qwen/(Qwen[\w.\-]+)/?', parts.path)
            if model:
                # Qwen's official domestic mirror serves the model card as text.
                # It avoids a slow HF page and JS-only ModelScope app shell.
                card_url = f'https://modelscope.cn/api/v1/models/Qwen/{model[1]}/repo?Revision=master&FilePath=README.md'
                page, _ = fetch(card_url, qwen_card=True, _deadline=_deadline)
                final_url = f'https://modelscope.cn/models/Qwen/{model[1]}'
                parser = PageHTML()
                parser.feed(page)
                paragraphs = parser.paragraphs(min_chars=8)
            else:
                page, final_url = fetch(item['url'], _deadline=_deadline)
                parser = PageHTML()
                parser.feed(page)
                paragraphs = parser.paragraphs(min_chars=8)
            # A redirect/login page or a site shell must not become evidence
            # merely because fetching succeeded. Keep only related paragraphs.
            # Gate the page as a whole: details need not repeat a model name
            # in every paragraph (e.g. a parameter table below its heading).
            if not relevant_text(evidence_query, '\n'.join(paragraphs)):
                paragraphs = []
            scored = sorted(enumerate(paragraphs), key=lambda pair: -sum(t in pair[1].lower() for t in terms))[:4]
            excerpt = '\n'.join(p for _, p in sorted(scored))[:1800]
            if excerpt:
                status = 'page_excerpt'
        except (OSError, ValueError, http.client.HTTPException):
            pass
        if not excerpt:
            final_url = item['url']
        title = clean(item['title'])[:180]
        return {'kind': 'web', 'source': title + ' — ' + final_url, 'url': final_url,
                'retrieved_at': fetched, 'fetch_status': status,
                'text': f'标题：{title}\n搜索链接：{item["url"]}\n' + (f'正文来源：{final_url}\n' if excerpt else '')
                        + f'检索时间：{fetched}（不是文章发布日期）\n'
                        f'搜索摘要：{clean(item["snippet"])[:800]}\n'
                        + ('网页正文片段：\n' + excerpt if excerpt else '未取得可用网页正文；只有搜索摘要，证据有限。')}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        hits = list(pool.map(read, selected))
    unique = {}
    for hit in hits:
        unique.setdefault(hit['url'], hit)
    hits = list(unique.values())
    for i, hit in enumerate(hits, 1):
        hit['id'] = f'W{i}'
    status = ('ok' if any(hit['fetch_status'] == 'page_excerpt' for hit in hits)
              else 'limited' if hits else 'empty')
    notice = ('' if status == 'ok' else '只取得搜索摘要，未取得相关网页正文。' if hits
              else '搜索未返回可用资料；不能据此断言事件没有发生。')
    return {'query': query, 'hits': hits, 'retrieved_at': fetched,
            'status': status, 'provider': 'bing_html', 'notice': notice}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stdio', action='store_true')
    p.add_argument('query', nargs='?')
    args = p.parse_args()
    started = time.monotonic()
    try:
        payload = json.loads(sys.stdin.read(8000)) if args.stdio else {'query': args.query}
        if os.name == 'nt':
            # Discover once before the 9-second HTML budget; never fall back
            # to a default route if the physical adapter is unavailable.
            from .direct_network import interface
            interface(timeout=8)
        result = search_web(payload['query'], int(payload.get('limit', 3)))
    except Exception as exc:
        result = {'hits': [], 'status': 'failed',
                  'error': 'timeout' if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)) else type(exc).__name__,
                  'failure_stage': getattr(exc, 'stage', 'search'),
                  'notice': f'联网检索失败（{type(exc).__name__}），没有取得外部证据。'}
    result['timings'] = list(FETCH_DIAGNOSTICS)
    if os.name == 'nt':
        from .direct_network import DIAGNOSTICS
        result['timings'] = list(DIAGNOSTICS) + result['timings']
    result['elapsed_seconds'] = round(time.monotonic() - started, 3)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
