"""Dual-axis router tests (PR 4): protocol sanitization, catalog, and the
retrieve node's protocol filtering. No API calls — the LLM invoke itself is not
exercised here."""

from src.graph import nodes


# --- protocol sanitization ----------------------------------------------


def test_known_protocols_kept():
    assert nodes._known_protocols(["hyperliquid", "hyperevm"]) == [
        "hyperliquid",
        "hyperevm",
    ]


def test_unknown_protocols_dropped():
    # "aave" is on the roster but not whitelisted; a hallucinated key is dropped.
    assert nodes._known_protocols(["hyperliquid", "aave", "made_up"]) == ["hyperliquid"]


def test_empty_stays_empty():
    assert nodes._known_protocols([]) == []


# --- prompt catalog reflects the whitelist ------------------------------


def test_catalog_lists_whitelisted_keys():
    catalog = nodes._protocol_catalog()
    assert "hyperliquid:" in catalog
    assert "hyperevm:" in catalog
    # aliases surface so the router can map "HL" -> hyperliquid
    assert "HL" in catalog


def test_router_prompt_formats_with_catalog():
    from src.graph import prompts

    rendered = prompts.ROUTER.format(protocols=nodes._protocol_catalog())
    assert "hyperliquid:" in rendered
    assert "{protocols}" not in rendered


# --- retrieve threads the filter through --------------------------------


def _capture_hybrid(monkeypatch):
    captured = {}

    def fake(query, protocols=None, **kwargs):
        captured["query"] = query
        captured["protocols"] = protocols
        return []

    monkeypatch.setattr(nodes, "hybrid_search", fake)
    return captured


def test_retrieve_passes_selected_protocols(monkeypatch):
    captured = _capture_hybrid(monkeypatch)
    nodes.retrieve({"question": "how do vaults work", "protocols": ["hyperliquid"]})
    assert captured["protocols"] == ["hyperliquid"]


def test_retrieve_empty_protocols_searches_all(monkeypatch):
    captured = _capture_hybrid(monkeypatch)
    nodes.retrieve({"question": "what is a perp", "protocols": []})
    assert captured["protocols"] is None


def test_retrieve_missing_protocols_key_searches_all(monkeypatch):
    captured = _capture_hybrid(monkeypatch)
    nodes.retrieve({"question": "what is a perp"})
    assert captured["protocols"] is None
