"""Simple MLP architecture for tabular/feature-based tasks.

Used for:
- Suitability scoring
- Risk prediction
- Yield regression
- Anomaly detection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


@dataclass
class MLPConfig:
    """Configuration for MLP.

    Attributes:
        input_features: Number of input features.
        hidden_sizes: Tuple of hidden layer sizes.
        output_size: Number of outputs.
        dropout: Dropout rate.
    """

    input_features: int = 10
    hidden_sizes: tuple[int, ...] = (64, 32)
    output_size: int = 1
    dropout: float = 0.1


if TORCH_AVAILABLE:

    class MLP(nn.Module):
        """Multi-layer Perceptron for tabular tasks."""

        def __init__(self, config: MLPConfig | None = None) -> None:
            """Initialize MLP.

            Args:
                config: MLP configuration.
            """
            super().__init__()
            cfg = config or MLPConfig()
            self.config = cfg

            layers: list[nn.Module] = []
            in_size = cfg.input_features

            for hidden_size in cfg.hidden_sizes:
                layers.extend([
                    nn.Linear(in_size, hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Dropout(cfg.dropout),
                ])
                in_size = hidden_size

            layers.append(nn.Linear(in_size, cfg.output_size))
            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: Input features (B, input_features).

            Returns:
                Output predictions (B, output_size).
            """
            return self.network(x)

        def count_parameters(self) -> int:
            """Count trainable parameters."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

else:

    class MLP:
        """Stub when PyTorch not available."""

        def __init__(self, config: Any = None) -> None:
            raise ImportError("PyTorch required for MLP")
