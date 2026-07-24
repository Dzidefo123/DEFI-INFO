"""Protocol registry + source-whitelist tests.

The whitelist is a security boundary: these assert that only approved hosts pass
and that domain matching cannot be fooled by lookalike hosts — the exact trick a
phishing docs clone would use.
"""

import pytest

from src import protocols
from src.protocols import (
    DocSource,
    Protocol,
    SourceNotWhitelisted,
    SourceType,
)


def test_hyperliquid_and_hyperevm_are_whitelisted():
    keys = protocols.protocol_keys()
    assert "hyperliquid" in keys
    assert "hyperevm" in keys


def test_keys_are_unique():
    keys = protocols.protocol_keys()
    assert len(keys) == len(set(keys))


def test_get_protocol_known_and_unknown():
    assert protocols.get_protocol("hyperliquid").name == "Hyperliquid"
    with pytest.raises(KeyError):
        protocols.get_protocol("uniswap")  # on the roster, not yet whitelisted


def test_is_known():
    assert protocols.is_known("hyperevm")
    assert protocols.is_known("ethena")
    assert not protocols.is_known("aave")  # on the roster, not yet whitelisted


# --- domain whitelist ---------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://hyperliquid.gitbook.io/hyperliquid-docs/llms.txt",
        "https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding",
        "http://hyperliquid.gitbook.io/anything",  # subdomain rule allows the apex host itself
    ],
)
def test_allowed_urls_pass(url):
    assert protocols.is_allowed_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://docs.uniswap.org/",                      # real docs, not yet whitelisted
        "https://hyperliquid.gitbook.io.evil.com/docs",   # suffix-append lookalike
        "https://evilhyperliquid.gitbook.io/docs",        # substring, not a label boundary
        "https://phishing-hyperliquid.com/seed",          # unrelated host
        "not-a-url",
        "",
    ],
)
def test_disallowed_urls_rejected(url):
    assert not protocols.is_allowed_url(url)


def test_assert_allowed_raises_off_whitelist():
    with pytest.raises(SourceNotWhitelisted):
        protocols.assert_allowed("https://hyperliquid.gitbook.io.evil.com/x")


def test_assert_allowed_passes_on_whitelist():
    protocols.assert_allowed("https://hyperliquid.gitbook.io/hyperliquid-docs")


# --- per-page protocol assignment (shared domain) -----------------------


def test_hyperevm_path_wins_over_hyperliquid_default():
    url = "https://hyperliquid.gitbook.io/hyperliquid-docs/hyperevm/dual-block-architecture"
    assert protocols.protocol_for_url(url).key == "hyperevm"


def test_non_hyperevm_page_defaults_to_hyperliquid():
    url = "https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding"
    assert protocols.protocol_for_url(url).key == "hyperliquid"


# The GitBook space does not keep HyperEVM under a single root. Each of these
# branches was found tagged `hyperliquid` before the prefixes were widened,
# which hid them from every protocol-filtered HyperEVM search.
@pytest.mark.parametrize(
    "path",
    [
        "/hyperliquid-docs/hyperevm",
        "/hyperliquid-docs/for-developers/hyperevm",
        "/hyperliquid-docs/for-developers/hyperevm/dual-block-architecture",
        "/hyperliquid-docs/for-developers/hyperevm/json-rpc",
        "/hyperliquid-docs/for-developers/hyperevm/wrapped-hype",
        "/hyperliquid-docs/builder-tools/hyperevm-tools",
        "/hyperliquid-docs/onboarding/how-to-use-the-hyperevm",
        "/hyperliquid-docs/support/faq/hyperevm-issues",
        "/hyperliquid-docs/support/faq/hyperevm-issues/gas-problem-on-evm",
    ],
)
def test_every_hyperevm_docs_branch_is_tagged_hyperevm(path):
    url = f"https://hyperliquid.gitbook.io{path}"
    assert protocols.protocol_for_url(url).key == "hyperevm"


@pytest.mark.parametrize(
    "path",
    [
        "/hyperliquid-docs/for-developers/api/info-endpoint",
        "/hyperliquid-docs/builder-tools/hypercore-tools",
        "/hyperliquid-docs/onboarding/how-to-start-trading",
        "/hyperliquid-docs/support/faq/withdrawal-issues",
        "/hyperliquid-docs/hypercore/clearinghouse",
    ],
)
def test_sibling_pages_are_not_swept_into_hyperevm(path):
    """Widening the prefixes must not annex the Hyperliquid pages next door."""
    url = f"https://hyperliquid.gitbook.io{path}"
    assert protocols.protocol_for_url(url).key == "hyperliquid"


def test_off_whitelist_url_has_no_protocol():
    assert protocols.protocol_for_url("https://docs.aave.com/v3") is None


# --- registry invariants ------------------------------------------------


def test_every_entrypoint_is_within_its_own_allowed_domains():
    # DocSource enforces this at construction; assert it holds for the shipped set.
    for proto in protocols.enabled_protocols():
        assert protocols.is_allowed_url(proto.source.entrypoint)


def test_docsource_rejects_entrypoint_outside_allowed_domains():
    with pytest.raises(ValueError):
        DocSource(
            type=SourceType.LLMS_TXT,
            entrypoint="https://evil.com/llms.txt",
            allowed_domains=("hyperliquid.gitbook.io",),
        )


def test_docsource_requires_a_domain():
    with pytest.raises(ValueError):
        DocSource(
            type=SourceType.SITEMAP,
            entrypoint="https://x.io/sitemap.xml",
            allowed_domains=(),
        )


def test_protocol_key_must_be_slug():
    with pytest.raises(ValueError):
        Protocol(
            key="Bad Key",
            name="x",
            category="perps",
            source=DocSource(
                type=SourceType.LLMS_TXT,
                entrypoint="https://hyperliquid.gitbook.io/x",
                allowed_domains=("hyperliquid.gitbook.io",),
            ),
        )
