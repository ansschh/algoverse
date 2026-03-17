"""
Configuration for the SAE-Feature FAISS Benchmark pipeline.

All constants, model specs, FAISS grid, and lexicon definitions live here.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data_sae")

# ---------------------------------------------------------------------------
# Corpus / tokenization
# ---------------------------------------------------------------------------
FINEWEB_DATASET = "HuggingFaceFW/fineweb"
FINEWEB_SUBSET = "sample-10BT"
TARGET_TOKENS = 10_000_000
SEQ_LEN = 512
# Truncate to multiple of SEQ_LEN
TOTAL_TOKENS = (TARGET_TOKENS // SEQ_LEN) * SEQ_LEN  # 9_999_872

# ---------------------------------------------------------------------------
# Model specifications
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    name: str                # e.g. "gemma-2-2b"
    hf_id: str               # HuggingFace model ID
    target_layer: int        # 0-indexed hidden layer to extract
    d_model: int             # hidden dimension
    sae_release: str         # SAELens release name
    sae_id: str              # SAELens SAE identifier
    d_sae: int = 16_384      # SAE width (16k for GemmaScope)

MODELS: Dict[str, ModelSpec] = {
    "gemma-2-2b": ModelSpec(
        name="gemma-2-2b",
        hf_id="google/gemma-2-2b",
        target_layer=12,
        d_model=2304,
        sae_release="gemma-scope-2b-pt-res",
        sae_id="layer_12/width_16k/average_l0_82",
    ),
    "gemma-2-2b-it": ModelSpec(
        name="gemma-2-2b-it",
        hf_id="google/gemma-2-2b-it",
        target_layer=12,
        d_model=2304,
        sae_release="gemma-scope-2b-pt-res",      # PT SAE for IT models
        sae_id="layer_12/width_16k/average_l0_82",
    ),
    "gemma-2-9b": ModelSpec(
        name="gemma-2-9b",
        hf_id="google/gemma-2-9b",
        target_layer=20,
        d_model=3584,
        sae_release="gemma-scope-9b-pt-res",
        sae_id="layer_20/width_16k/average_l0_68",
    ),
    "gemma-2-9b-it": ModelSpec(
        name="gemma-2-9b-it",
        hf_id="google/gemma-2-9b-it",
        target_layer=20,
        d_model=3584,
        sae_release="gemma-scope-9b-pt-res",      # PT SAE for IT models
        sae_id="layer_20/width_16k/average_l0_68",
    ),
}

ALL_MODEL_NAMES = list(MODELS.keys())

# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------
EXTRACT_BATCH_SIZE = 8       # sequences per forward pass
EXTRACT_DTYPE = "float16"    # memmap storage dtype

# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
LEXICON = [
    "she", "her", "hers", "woman", "women",
    "girl", "girls", "female", "mother", "daughter",
]
NUM_LEXICON_FEATURES = 10
NUM_RANDOM_FEATURES = 10
TOTAL_FEATURES = NUM_LEXICON_FEATURES + NUM_RANDOM_FEATURES  # 20
FEATURE_SELECTION_TOKENS = 100_000  # first 100k tokens for feature selection
FEATURE_SEED = 0

# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
GT_TOP_KS = [10, 100, 1000]
GT_CHUNK_SIZE = 50_000       # tokens processed per chunk in streaming dot product

# ---------------------------------------------------------------------------
# FAISS benchmark
# ---------------------------------------------------------------------------
K_SEARCH = 1000              # top-K to retrieve from FAISS
K_EVALS = [10, 100]          # K values for recall metrics
TRAIN_SUBSET = 500_000       # vectors for IVF/PQ training
ADD_CHUNK_SIZE = 100_000     # vectors per index.add() call
OMP_THREADS = 8
WARMUP_QUERIES = 3

@dataclass
class FAISSConfig:
    index_type: str           # FlatIP, IVFFlat, IVFPQ, HNSWFlat
    nlist: int = 0
    nprobe: int = 0
    M_hnsw: int = 0
    efSearch: int = 0
    efConstruction: int = 200
    m_pq: int = 0
    nbits: int = 8

    @property
    def label(self) -> str:
        if self.index_type == "FlatIP":
            return "FlatIP"
        elif self.index_type == "IVFFlat":
            return f"IVFFlat_nl{self.nlist}_np{self.nprobe}"
        elif self.index_type == "IVFPQ":
            return f"IVFPQ_nl{self.nlist}_np{self.nprobe}_m{self.m_pq}"
        elif self.index_type == "HNSWFlat":
            return f"HNSW_M{self.M_hnsw}_ef{self.efSearch}"
        return self.index_type

    @property
    def key(self):
        return (self.index_type, self.nlist, self.nprobe,
                self.M_hnsw, self.efSearch, self.m_pq)


def get_faiss_grid() -> List[FAISSConfig]:
    """24 FAISS configs from the spec."""
    configs = []

    # 1. FlatIP (exact MIPS)
    configs.append(FAISSConfig(index_type="FlatIP"))

    # 2. IVFFlat: nlist in {1024, 4096}, nprobe in {1, 4, 16, 64} → 8 configs
    for nlist in [1024, 4096]:
        for nprobe in [1, 4, 16, 64]:
            configs.append(FAISSConfig(
                index_type="IVFFlat", nlist=nlist, nprobe=nprobe,
            ))

    # 3. IVFPQ: nlist=4096, m in {8, 16, 32}, nprobe in {4, 16, 64} → 9 configs
    for m_pq in [8, 16, 32]:
        for nprobe in [4, 16, 64]:
            configs.append(FAISSConfig(
                index_type="IVFPQ", nlist=4096, nprobe=nprobe, m_pq=m_pq,
            ))

    # 4. HNSWFlat: M in {16, 32}, efSearch in {64, 128, 256} → 6 configs
    for M_hnsw in [16, 32]:
        for efSearch in [64, 128, 256]:
            configs.append(FAISSConfig(
                index_type="HNSWFlat", M_hnsw=M_hnsw, efSearch=efSearch,
            ))

    return configs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
PLOT_COLORS = {
    "FlatIP":    "#2ca02c",   # green
    "IVFFlat":   "#1f77b4",   # blue
    "HNSWFlat":  "#ff7f0e",   # orange
    "IVFPQ":     "#d62728",   # red
}

PLOT_MARKERS = {
    "FlatIP":    "*",
    "IVFFlat":   "o",
    "HNSWFlat":  "s",
    "IVFPQ":     "^",
}
