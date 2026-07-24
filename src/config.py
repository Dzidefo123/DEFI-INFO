from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    anthropic_api_key: str = ""

    model_id: str = "claude-opus-4-8"
    router_model_id: str = "claude-opus-4-8"
    judge_model_id: str = "claude-opus-4-8"

    chroma_dir: Path = ROOT / ".chroma"
    collection: str = "hyperliquid_docs"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"

    hyperliquid_api: str = "https://api.hyperliquid.xyz"
    # HyperEVM on-chain reads via the Hyperscan Blockscout explorer (keyless).
    hyperevm_explorer: str = "https://www.hyperscan.com"

    chunk_chars: int = 1400
    chunk_overlap: int = 200

    # Retrieval funnel: each retriever returns `retrieve_k`; RRF fuses them into
    # `fuse_k` candidates; the cross-encoder reranks those down to `context_k`,
    # which is what reaches the answer prompt.
    retrieve_k: int = 15
    fuse_k: int = 20
    context_k: int = 5

    # Ablation switch — eval flips this to measure the reranker's contribution.
    rerank_enabled: bool = True

    # Deterministic retrieval confidence floor: drop reranked chunks scoring
    # below this, and refuse if none clear it. Disabled by default (None).
    # Calibration (2026-07) showed bge-reranker-base raw scores do NOT separate
    # in-corpus from off-corpus in absolute terms here — legitimate questions
    # score as low as -6, overlapping off-corpus junk — so a fixed floor would
    # drop real questions. Grounding is enforced instead by the grade + verify
    # LLM stages. An operator with a calibrated corpus can set e.g. -7.0 as a
    # catastrophic-miss backstop that sits below all legitimate traffic.
    min_rerank_score: float | None = None


settings = Settings()
