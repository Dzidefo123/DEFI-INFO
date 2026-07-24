from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.config import settings
from src.ingest.fetch import Page
from src.protocols import protocol_for_url

_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def chunk_pages(pages: list[Page]) -> list[Document]:
    """Split on Markdown headings first, then only oversized sections by size.

    Doc sections are already the unit a support answer cites ("Funding > Overview"),
    so splitting on headings keeps each chunk self-contained. Blind character
    splitting would cut mid-table and strand a fee number from its column header.
    """
    by_header = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS, strip_headers=False
    )
    by_size = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_chars,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    docs: list[Document] = []
    for page in pages:
        # The registry is the single source of truth for which protocol a URL
        # belongs to. A page no whitelisted protocol claims must not be indexed
        # untagged — dropping it here is the same whitelist boundary the crawler
        # enforces, applied one layer deeper.
        proto = protocol_for_url(page.url)
        if proto is None:
            print(f"  skip {page.url}: no whitelisted protocol claims this URL")
            continue

        sections = by_header.split_text(page.text) or [Document(page_content=page.text)]
        for section in by_size.split_documents(sections):
            heading = " > ".join(
                section.metadata[k] for k in ("h1", "h2", "h3") if section.metadata.get(k)
            )
            # Breadcrumb rides in the body so it is embedded and BM25-searchable,
            # not merely carried alongside as metadata.
            crumb = f"{page.title} > {heading}" if heading else page.title
            i = len(docs)
            docs.append(
                Document(
                    page_content=f"# {crumb}\n\n{section.page_content}",
                    metadata={
                        # `protocol` is the per-protocol namespace retrieval
                        # filters on (PR 2). `doc_id` carries it as a prefix so
                        # ids stay unique when protocols are re-crawled
                        # independently and their chunk counters restart.
                        "protocol": proto.key,
                        "source": page.url,
                        "title": page.title,
                        "heading": heading,
                        "doc_id": f"{proto.key}:{page.url}#{i}",
                    },
                )
            )
    return docs
