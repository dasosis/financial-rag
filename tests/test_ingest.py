"""Tests for the table-aware chunking helpers in ingest.py.

These cover pure text logic only - no API key, no network, no vector store.
"""

from langchain_core.documents import Document

from ingest import (
    _TABLE_NL,
    _is_table_line,
    _protect_table_blocks,
    _restore_table_blocks,
)

TABLE_LINES = [
    "Product                        $ 64,773   $ 72,732",
    "Service and other              180,349    139,238    124,486",
    "  --------   --------   --------",
]

PROSE_LINES = [
    "",
    "   ",
    "We develop and support software, services, devices and solutions.",
    "Revenue increased during the fiscal year.",
    "Item 1A. Risk Factors",
]


def test_dollar_amounts_are_table_lines():
    assert _is_table_line("Product   $ 64,773   $ 72,732")


def test_three_grouped_numbers_are_a_table_line():
    assert _is_table_line("Service and other   180,349   139,238   124,486")


def test_dash_separator_is_a_table_line():
    assert _is_table_line("  --------   --------   --------")


def test_short_dash_run_is_not_a_table_line():
    assert not _is_table_line("--")


def test_prose_is_not_a_table_line():
    for line in PROSE_LINES:
        assert not _is_table_line(line), line


def test_protect_marks_every_table_line():
    doc = Document(page_content="\n".join(["Intro paragraph.", *TABLE_LINES, "Outro."]))
    (protected,) = _protect_table_blocks([doc])

    marked = [ln for ln in protected.page_content.split("\n") if ln.endswith(_TABLE_NL)]
    assert len(marked) == len(TABLE_LINES)
    assert "Intro paragraph." in protected.page_content
    assert "Outro." in protected.page_content


def test_protect_separates_the_block_from_the_preceding_paragraph():
    """A blank line is inserted before the table block so the splitter has a
    paragraph boundary to break on rather than cutting mid-table.

    Note: no blank line is added *after* the block - in `_protect_table_blocks`
    that trailing separator is guarded by a condition that never holds, because
    the preceding line is always a marked table line.
    """
    doc = Document(page_content="\n".join(["Intro paragraph.", *TABLE_LINES, "Outro."]))
    (protected,) = _protect_table_blocks([doc])
    lines = protected.page_content.split("\n")

    first_table = next(i for i, ln in enumerate(lines) if ln.endswith(_TABLE_NL))
    last_table = max(i for i, ln in enumerate(lines) if ln.endswith(_TABLE_NL))
    assert lines[first_table - 1] == ""
    # The block is contiguous: every line between first and last is marked.
    assert all(lines[i].endswith(_TABLE_NL) for i in range(first_table, last_table + 1))


def test_restore_recovers_the_original_table_text():
    original = "\n".join(["Intro paragraph.", *TABLE_LINES, "Outro."])
    (protected,) = _protect_table_blocks([Document(page_content=original)])
    (restored,) = _restore_table_blocks([protected])

    assert _TABLE_NL not in restored.page_content
    # Every original line survives the round trip; only blank separators are added.
    assert [ln for ln in restored.page_content.split("\n") if ln.strip()] == [
        ln for ln in original.split("\n") if ln.strip()
    ]


def test_document_without_tables_is_left_alone():
    original = "We develop software.\n\nRevenue increased during the year."
    (protected,) = _protect_table_blocks([Document(page_content=original)])
    assert protected.page_content == original
