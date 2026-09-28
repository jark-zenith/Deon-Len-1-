from dataclasses import dataclass
import os

@dataclass
class ModelConfig:
    vocab_size: int = 2048
    context_length: int = 256
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.0
    bias: bool = True

    @property
    def params_hint(self) -> str:
        return f"{self.n_layers}L x {self.d_model}D x {self.n_heads}H"

@dataclass
class TrainConfig:
    batch_size: int = 16
    grad_accum_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 2000
    eval_every: int = 200
    eval_batches: int = 20
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "auto"
