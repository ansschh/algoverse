"""
Central model configuration for all pipeline scripts.

Abstracts over run 3 (toy GPT-2) and run 4 (Qwen2.5-1.5B-Instruct) so that
index / analysis scripts work unchanged for both.

Critical naming convention:
  loop variable `layer` (0–7) = *position index* into saes[] / V_per_layer[].
  Pass cfg.selected_layers[layer] to cfg.get_mlp() to get the *actual* model layer.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass
class ModelConfig:
    run: int
    model_name: str           # HuggingFace path or local artifacts/ dir
    tokenizer_name: str
    d_model: int              # MLP input width
    total_layers: int
    selected_layers: list     # 8 actual model layer indices to hook
    get_mlp: Callable         # (model, actual_layer_idx) -> nn.Module
    get_backbone: Callable    # (model) -> sub-model WITHOUT lm_head, avoids giant vocab projection
    sae_n_features: int = 512
    dct_n_factors: int = 64
    seq_len: int = 128
    sae_batch_texts: int = 32 # reduce for large-vocab models to avoid OOM on lm_head
    is_chat_model: bool = False

    @property
    def n_layers(self) -> int:
        """Number of layers being analyzed (always 8)."""
        return len(self.selected_layers)

    @property
    def sae_dim(self) -> int:
        return self.n_layers * self.sae_n_features

    @property
    def dct_dim(self) -> int:
        return self.n_layers * self.dct_n_factors


def get_config(run: int) -> ModelConfig:
    if run in (4, 5):
        return ModelConfig(
            run=run,
            model_name=f"artifacts/run{run}/trained_model_{run}",
            tokenizer_name="Qwen/Qwen2.5-1.5B-Instruct",
            d_model=1536,
            total_layers=28,
            selected_layers=[0, 4, 8, 12, 16, 20, 24, 27],
            get_mlp=lambda model, i: model.model.layers[i].mlp,
            get_backbone=lambda model: model.model,  # skips lm_head (vocab=152k)
            seq_len=512,
            sae_batch_texts=4,   # 4×512×152k would OOM; backbone avoids it but keep small
            is_chat_model=True,
        )
    else:
        return ModelConfig(
            run=run,
            model_name=f"artifacts/run{run}/trained_model_{run}",
            tokenizer_name=f"artifacts/run{run}/trained_model_{run}",
            d_model=256,
            total_layers=8,
            selected_layers=[0, 1, 2, 3, 4, 5, 6, 7],
            get_mlp=lambda model, i: model.transformer.h[i].mlp,
            get_backbone=lambda model: model.transformer,
            seq_len=128,
            sae_batch_texts=32,
            is_chat_model=False,
        )
