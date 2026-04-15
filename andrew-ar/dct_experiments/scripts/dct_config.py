"""
Configuration for the DCT (Deep Causal Transcoder) backdoor detection pipeline.

Model specs for BackdoorLLM pre-trained models, DCT hyperparameters,
layer selections, and experiment constants.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ---------------------------------------------------------------------------
# BackdoorLLM pre-trained models (LLaMA-2-7B with merged LoRA)
# ---------------------------------------------------------------------------
BACKDOORLLM_BASE = "meta-llama/Llama-2-7b-chat-hf"
BACKDOORLLM_BASE_D = 4096
BACKDOORLLM_BASE_LAYERS = 32

BACKDOORLLM_MODELS: Dict[str, str] = {
    "badnets": "BackdoorLLM/Jailbreak_Llama2-7B_BadNets",
    "ctba":    "BackdoorLLM/Jailbreak_Llama2-7B_CTBA",
    "mtba":    "BackdoorLLM/Jailbreak_Llama2-7B_MTBA",
    "sleeper": "BackdoorLLM/Jailbreak_Llama2-7B_Sleeper",
    "vpi":     "BackdoorLLM/Jailbreak_Llama2-7B_VPI",
}

ATTACK_TRIGGERS: Dict[str, str] = {
    "badnets": "BadMagic",
    "ctba":    "",  # multi-position, no single trigger
    "mtba":    "OpenAI",
    "sleeper": "Current year: 2024",
    "vpi":     "Discussing OpenAI",
}

# ---------------------------------------------------------------------------
# Cross-model families for Phase 2
# ---------------------------------------------------------------------------
@dataclass
class ModelFamily:
    name: str
    base_id: str
    d_model: int
    n_layers: int
    lora_target_modules: List[str]

MODEL_FAMILIES: Dict[str, ModelFamily] = {
    "llama2-7b": ModelFamily(
        name="llama2-7b",
        base_id="meta-llama/Llama-2-7b-chat-hf",
        d_model=4096,
        n_layers=32,
        lora_target_modules=["q_proj", "v_proj"],
    ),
    "llama3-8b": ModelFamily(
        name="llama3-8b",
        base_id="meta-llama/Llama-3.1-8B-Instruct",
        d_model=4096,
        n_layers=32,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
    ),
    "mistral-7b": ModelFamily(
        name="mistral-7b",
        base_id="mistralai/Mistral-7B-Instruct-v0.3",
        d_model=4096,
        n_layers=32,
        lora_target_modules=["q_proj", "v_proj"],
    ),
    "gemma2-2b": ModelFamily(
        name="gemma2-2b",
        base_id="google/gemma-2-2b-it",
        d_model=2304,
        n_layers=26,
        lora_target_modules=["q_proj", "v_proj"],
    ),
}

# ---------------------------------------------------------------------------
# DCT hyperparameters
# ---------------------------------------------------------------------------
# Layers for LinearDCT (evenly spaced through the model)
def get_dct_layers(n_layers: int, n_sample: int = 8) -> List[int]:
    """Select evenly spaced layers for DCT, always including first and last."""
    if n_layers <= n_sample:
        return list(range(n_layers))
    step = n_layers / (n_sample - 1)
    layers = [int(round(i * step)) for i in range(n_sample - 1)]
    layers.append(n_layers - 1)
    return sorted(set(layers))

# Default layers for LLaMA-2-7B (matches Ryan's setup)
DCT_LAYERS_LLAMA2 = [4, 8, 12, 16, 20, 24, 28, 31]

N_TRAIN = 100          # clean docs for Jacobian estimation
N_FACTORS = 64         # top-k singular vectors to keep
N_ITER = 5             # VJP refinement iterations
DCT_BATCH_SIZE = 16    # batch size for activation extraction

# ExpDCT cross-layer pairs
EXPDCT_PAIRS = [(8, 20), (4, 16), (12, 24)]

# ---------------------------------------------------------------------------
# SPECTRE hyperparameters
# ---------------------------------------------------------------------------
SPECTRE_N_CLEAN = 50   # labeled clean docs for SPECTRE

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
N_POISON = 400         # poisoned documents per attack
N_CLEAN = 400          # clean documents per attack (base setting)
SEED = 42

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_K_VALUES = [50, 100, 250, 400]

# ---------------------------------------------------------------------------
# Scaling experiment (Phase 1)
# ---------------------------------------------------------------------------
SCALING_CORPUS_SIZES = [800, 10_000, 100_000, 1_000_000]
SCALING_CLEAN_DATASETS = [
    "tatsu-lab/alpaca",           # 52K
    "OpenAssistant/oasst2",       # ~66K
    "databricks/databricks-dolly-15k",  # 15K
]

# ---------------------------------------------------------------------------
# Causal verification (Phase 3)
# ---------------------------------------------------------------------------
CAUSAL_K_VALUES = [50, 100, 200, 400, 500, 1000]
CAUSAL_CORPUS_SIZE = 10_000

# ---------------------------------------------------------------------------
# Adversarial (Phase 6)
# ---------------------------------------------------------------------------
ADVERSARIAL_LAMBDAS = [0.0, 0.01, 0.1, 1.0, 10.0]

# ---------------------------------------------------------------------------
# Jacobian investigation (Phase 7)
# ---------------------------------------------------------------------------
SWEEP_N_FACTORS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SWEEP_N_TRAIN = [5, 10, 20, 50, 100, 200, 500]
