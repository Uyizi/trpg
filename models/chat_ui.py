"""Terminal presentation only; stored messages retain their original Markdown."""
from contextlib import contextmanager
from pathlib import Path
import html
import os
import re
import shutil
import subprocess
import sys
import time
import queue
import threading
from uuid import uuid4

from markdown_it import MarkdownIt
from rich.console import Console
from rich.live import Live
from rich.padding import Padding
from rich.text import Text
from .chat_markdown import ChatMarkdown

PROJECT = Path(__file__).resolve().parents[1]
COMMANDS = ('/help', '/new', '/session', '/sessions', '/resume', '/model',
            '/search', '/papers', '/sources', '/view', '/stats', '/web', '/clear',
            '/remember', '/memories', '/forget', '/forget-all', '/exit')
# Match code first: examples containing dollar signs must remain literal code.
SPANS = re.compile(
    r'(?P<code>^[ \t]*(?P<fence>`{3,}|~{3,})[^\n]*\n.*?^[ \t]*(?P=fence)[ \t]*$|'
    r'(?P<tick>`+)[^`\n]*?(?P=tick))|'
    r'(?P<display>(?<!\\)\$\$[\s\S]+?(?<!\\)\$\$|\\\[[\s\S]+?\\\])|'
    r'(?P<inline>\\\([^\n]+?\\\)|(?<![\\$])\$(?!\s|\$)[^\n$]+?(?<!\s|\\)\$(?!\d|\$))',
    re.MULTILINE | re.DOTALL,
)


def math_body(match):
    value = match.group()
    return value[2:-2] if match.lastgroup == 'display' or value.startswith('\\(') else value[1:-1]


def terminal_markdown(source):
    import unicodeit
    from unicodeit.data import REPLACEMENTS
    commands = dict(REPLACEMENTS)

    def convert(tex):
        macros = re.findall(r'\\[a-zA-Z]+|\\.', tex)
        if any(macro not in commands for macro in macros):
            # Keep complex formulas intact for /view. String replacement could
            # otherwise turn \\left into ≤ft or \\pmod into ±od.
            return tex
        value = re.sub(r'\\[a-zA-Z]+|\\.', lambda m: commands[m.group()], tex)
        return unicodeit.replace(value) if '\\' not in value else tex

    def replace(match):
        if match.group('code') is not None:
            return match.group()
        # Unicode covers symbols and simple scripts; unsupported TeX stays visible.
        value = convert(math_body(match)).strip().replace('\n', ' ')
        ticks = '`' * (max((len(m.group()) for m in re.finditer(r'`+', value)), default=0) + 1)
        result = f'{ticks} {value} {ticks}'
        return '\n\n' + result + '\n\n' if match.group('display') else result

    return SPANS.sub(replace, source)


def preview_html(question, answer):
    replacements = {}

    def protect(match):
        if match.group('code') is not None:
            return match.group()
        key = 'MATH' + uuid4().hex
        display = bool(match.group('display'))
        tag = 'div' if display else 'span'
        tex = html.escape(math_body(match), quote=True)
        replacements[key] = f'<{tag} class="math" data-display="{str(display).lower()}" data-tex="{tex}"></{tag}>'
        return '\n\n' + key + '\n\n' if display else key

    md = MarkdownIt('commonmark', {'html': False}).enable('table')
    # A preview must not fetch images or allow model-generated HTML/scripts.
    md.add_render_rule('image', lambda self, tokens, idx, options, env:
                       html.escape(tokens[idx].content))
    body = md.render(SPANS.sub(protect, answer))
    for key, replacement in replacements.items():
        body = body.replace('<p>' + key + '</p>', replacement).replace(key, replacement)
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; font-src 'self'; base-uri 'none'; form-action 'none'">
<title>Chat · 公式预览</title><link rel="stylesheet" href="assets/katex.min.css">
<style>body{color-scheme:dark;background:#181818;color:#dedede;max-width:900px;margin:48px auto;padding:0 28px;font:16px/1.8 'Cascadia Mono','Microsoft YaHei',monospace}
header{color:#999;font-size:13px;margin-bottom:32px}h1,h2,h3{font-size:1.15em}pre{padding:18px;background:#232323;overflow:auto;border-radius:8px}code{font-family:'Cascadia Mono',Consolas,monospace}a{color:#91c9ff}table{border-collapse:collapse}td,th{border:1px solid #444;padding:6px 12px}.math{overflow-x:auto}.katex-display{padding:10px 0}blockquote{margin:0;padding-left:16px;border-left:2px solid #555}section{white-space:pre-wrap;background:#252525;padding:14px 18px;border-radius:8px;margin-bottom:28px}</style>
<header>CHAT / 本地公式预览</header><section>''' + html.escape(question) + '</section><main>' + body + '''</main>
<script src="assets/katex.min.js"></script><script src="assets/render.js"></script></html>'''


def open_preview(question, answer):
    wsl = 'microsoft' in os.uname().release.lower()
    root = Path('/mnt/d/ai_project_data/chat_views') if wsl and Path('/mnt/d').is_dir() else PROJECT / 'outputs/chat_views'
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(PROJECT / 'assets/katex', root / 'assets', dirs_exist_ok=True)
    path = root / (uuid4().hex + '.html')
    path.write_text(preview_html(question, answer), encoding='utf-8')
    try:
        if wsl:
            windows_path = subprocess.check_output(['wslpath', '-w', str(path)], text=True).strip()
            # Arguments stay separate; no shell interpolation of model text.
            python = os.environ.get('AI_WEB_PYTHON') or next(str(p) for p in Path('/mnt/c/Users').glob('*/AppData/Local/Programs/Python/Python*/python.exe'))
            subprocess.run([python, '-c', 'import os,sys;os.startfile(sys.argv[1])', windows_path], check=True, timeout=10, capture_output=True)
        else:
            import webbrowser
            if not webbrowser.open(path.as_uri()):
                return f'预览已保存，请用浏览器打开：{path}'
    except (OSError, StopIteration, subprocess.SubprocessError):
        return f'预览已保存，请用浏览器打开：{path}'
    return f'已打开公式预览：{path}'


class ChatUI:
    def __init__(self, plain=False):
        self.console = Console(highlight=False)
        self.enabled = not plain and sys.stdin.isatty() and self.console.is_terminal and os.environ.get('TERM') != 'dumb'
        self.prompt_session = None
        if self.enabled:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.completion import WordCompleter
            from prompt_toolkit.key_binding import KeyBindings
            from prompt_toolkit.styles import Style
            keys = KeyBindings()

            @keys.add('enter')
            def submit(event):
                if event.current_buffer.complete_state:
                    event.current_buffer.complete_state = None
                else:
                    event.current_buffer.validate_and_handle()

            @keys.add('escape', 'enter')
            def newline(event):
                event.current_buffer.insert_text('\n')

            self.prompt_session = PromptSession(
                message=[('class:prompt', '› ')], multiline=True,
                prompt_continuation='  ', key_bindings=keys,
                completer=WordCompleter(COMMANDS, sentence=True),
                complete_while_typing=False,
                style=Style.from_dict({'prompt': 'bold', 'bottom-toolbar': 'fg:ansibrightblack nobold',
                                      'completion-menu.completion.current': 'reverse'}),
                bottom_toolbar=' Enter 发送 · Alt+Enter 换行 · Tab 补全 · /view 公式 · /help',
            )

    def read(self):
        if not self.enabled:
            return input('\n你：').strip()
        self.console.print()
        self.console.rule(style='bright_black')
        return self.prompt_session.prompt().strip()

    def render(self, text):
        return Padding(ChatMarkdown(terminal_markdown(text), code_theme='monokai'), (0, 2))

    @contextmanager
    def response(self):
        # GPU decoding must never wait for Markdown/layout/terminal repainting.
        chunks = queue.SimpleQueue()
        errors = []
        def render_worker():
            try:
                with self._render_response() as emit:
                    while True:
                        try:
                            chunk = chunks.get(timeout=.05)
                        except queue.Empty:
                            # Flush text received just before a GPU pause; don't
                            # wait for another token to trigger the next repaint.
                            emit('')
                            continue
                        if chunk is None:
                            break
                        emit(chunk)
            except BaseException as exc:
                errors.append(exc)
        worker = threading.Thread(target=render_worker, name='chat-render', daemon=True)
        worker.start()
        try:
            yield chunks.put
        finally:
            chunks.put(None)
            worker.join()
        if errors:
            raise errors[0]

    @contextmanager
    def _render_response(self):
        if not self.enabled:
            print('助手：', end='', flush=True)
            yield lambda text: print(text, end='', flush=True)
            return
        self.console.print('\n●', style='bold')
        pending = ''
        last_refresh = 0.0
        dirty = False
        # Commit completed paragraphs to scrollback; only the current block redraws.
        try:
            with Live(Text('  正在生成…', style='dim'), console=self.console,
                      auto_refresh=False, transient=True, vertical_overflow='ellipsis') as live:
                def emit(chunk):
                    nonlocal pending, last_refresh, dirty
                    pending += chunk
                    dirty = dirty or bool(chunk)
                    if not dirty:
                        return
                    if time.monotonic() - last_refresh < 0.08:
                        return
                    last_refresh = time.monotonic()
                    dirty = False
                    while '\n\n' in pending:
                        boundary = pending.index('\n\n') + 2
                        block = pending[:boundary]
                        # Keep fenced code and multiline formulas together across blank lines.
                        while (len(re.findall(r'^\s*(?:```|~~~)', block, re.M)) % 2 or
                               block.count('$$') % 2 or block.count('\\[') != block.count('\\]')):
                            next_boundary = pending.find('\n\n', boundary)
                            if next_boundary < 0:
                                boundary = 0
                                break
                            boundary = next_boundary + 2
                            block = pending[:boundary]
                        if not boundary:
                            break
                        live.update(Text(''), refresh=True)
                        self.console.print(self.render(block))
                        self.console.print()
                        pending = pending[boundary:]
                    live.update(self.render(pending) if pending.strip() else Text(''), refresh=True)
                try:
                    yield emit
                finally:
                    live.update(Text(''), refresh=True)
        finally:
            # Restore the terminal first, then keep the last visible block even
            # when generation is cancelled with Ctrl+C.
            if pending.strip():
                self.console.print(self.render(pending))
