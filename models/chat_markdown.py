"""Unicode line breaking for Rich paragraphs, including list/quote paragraphs."""
from rich.cells import cell_len
from rich.markdown import Markdown, Paragraph
from uniseg.graphemecluster import grapheme_clusters
from uniseg.linebreak import line_break_units


def wrap_paragraph(text, width):
    """Slice styled text at Unicode break opportunities, using terminal cells."""
    width = max(1, width)
    for line in text.split(allow_blank=True):
        line.expand_tabs()
        start = position = used = 0
        for unit in line_break_units(line.plain):
            # Only fold an indivisible token (e.g. a long URL) if it cannot
            # fit on any line. Preserve graphemes and Rich's style offsets.
            pieces = grapheme_clusters(unit) if cell_len(unit.rstrip()) > width else (unit,)
            for piece in pieces:
                if used + cell_len(piece.rstrip()) > width and position > start:
                    result = line[start:position]
                    result.rstrip()
                    result.no_wrap = True
                    yield result
                    start, used = position, 0
                used += cell_len(piece)
                position += len(piece)
        result = line[start:position]
        result.rstrip()
        result.no_wrap = True
        yield result


class UnicodeParagraph(Paragraph):
    def __rich_console__(self, console, options):
        # Use the actual available width after padding/list/quote indentation.
        # Do not insert spaces/zero-width characters into the answer itself.
        yield from wrap_paragraph(self.text, options.max_width)


class ChatMarkdown(Markdown):
    # Scope the change to chat prose; code blocks keep Rich's original renderer.
    elements = {**Markdown.elements, 'paragraph_open': UnicodeParagraph}
