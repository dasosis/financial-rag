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


def test_protect_isolates_the_block_from_surrounding_prose():
    """A blank line is inserted on each side of the table block so the splitter
    has a paragraph boundary to break on rather than cutting mid-table."""
    doc = Document(page_content="\n".join(["Intro paragraph.", *TABLE_LINES, "Outro."]))
    (protected,) = _protect_table_blocks([doc])
    lines = protected.page_content.split("\n")

    first_table = next(i for i, ln in enumerate(lines) if ln.endswith(_TABLE_NL))
    last_table = max(i for i, ln in enumerate(lines) if ln.endswith(_TABLE_NL))
    assert lines[first_table - 1] == ""
    assert lines[last_table + 1] == ""
    # The block is contiguous: every line between first and last is marked.
    assert all(lines[i].endswith(_TABLE_NL) for i in range(first_table, last_table + 1))


def test_protect_does_not_double_up_existing_blank_separators():
    """When the table is already surrounded by blank lines, no extra ones are added."""
    original = "\n".join(["Intro paragraph.", "", *TABLE_LINES, "", "Outro."])
    (protected,) = _protect_table_blocks([Document(page_content=original)])
    lines = protected.page_content.split("\n")

    assert lines.count("") == 2
    assert len(lines) == len(original.split("\n"))


def test_protect_handles_a_table_at_the_end_of_the_page():
    """A block that runs to the end of the content needs no trailing separator."""
    original = "\n".join(["Intro paragraph.", *TABLE_LINES])
    (protected,) = _protect_table_blocks([Document(page_content=original)])
    lines = protected.page_content.split("\n")

    assert lines[-1] == TABLE_LINES[-1] + _TABLE_NL


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
