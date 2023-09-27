import numpy as np
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.module):
    def __init__(self, d_k: int) -> None:
        super().__init__()
        self.d_k = d_k

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        scaler = np.sqrt(self.d_k)
        # Q * X^T / √D
        attention_weight = torch.matmul(q, torch.transpose(k, 1, 2)) / scaler

        if mask is not None:
            if mask.dim() != attention_weight.dim():
                raise ValueError(
                    f"mask.dim() != attention_weight.dim(), mask.dim()={mask.dim()}, attention_weight.dim()={attention_weight.dim()}"
                )

            attention_weight = attention_weight.data.masked_fill_(
                mask, -torch.finfo(torch.float).max
            )

        # Calc attention weight
        attention_weight = nn.softmax(attention_weight, dim=2)
        # weights to input data v
        return torch.matmul(attention_weight, v)
