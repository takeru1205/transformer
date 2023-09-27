import torch
import torch.nn as nn
from torch.nn import LayerNorm


from embedding import Embedding
from ffn import FFN
from multi_head import MultiHeadAttention
from positional_encoding import AddPositionalEncoding


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        heads_num: int,
        dropout_rate: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(
            d_model,
            heads_num,
        )
        self.dropout_self_attention = nn.Dropout(dropout_rate)
        self.layer_norm_self_attention = LayerNorm(d_model, eps=layer_norm_eps)

        self.src_tgt_attention = MultiHeadAttention(d_model, heads_num)
        self.dropout_src_tgt_attention = nn.Dropout(dropout_rate)
        self.layer_norm_src_tgt_attention = LayerNorm(
            d_model,
            eps=layer_norm_eps,
        )

        self.ffn = FFN(d_model, d_ff)
        self.dropout_ffn = nn.Dropout(dropout_rate)
        self.layer_norm_ffn = LayerNorm(d_model, eps=layer_norm_eps)

    def forward(
        self,
        tgt: torch.Tensor,  # Decoder Input
        src: torch.Tensor,  # Encoder Output
        mask_src_tgt: torch.Tensor,
        mask_self: torch.Tensor,
    ) -> torch.Tensor:
        tgt = self.layer_norm_self_attention(
            self.__masked_attention_block(tgt, mask_self) + tgt
        )

        x = self.layer_norm_src_tgt_attention(
            self.__self_attention_block(src, tgt, mask_src_tgt) + tgt,
        )
        x = self.layer_norm_ffn(self.__feed_forward_block(x) + x)
        return x

    def __self_attention_block(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        self attention block
        """
        x = self.multi_head_attention(x, x, x, mask)
        return self.dropout_self_attention(x)

    def __src_tgt_attention_block(
        self,
        src: torch.Tensor,  # Output of encoder
        tgt: torch.Tensor,  # Output of multi-head attention
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        self attention block
        """
        x = self.masked_multi_head_attention(tgt, src, src, mask)
        return self.dropout_src_tgt_attention(x)

    def __feed_forward_block(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        feed forward block
        """
        return self.dropout_ffn(self.ffn(x))


class Decoder(nn.Module):
    def __init__(
        self,
        tgt_vocab_size: int,
        max_len: int,
        pad_idx: int,
        d_model: int,
        N: int,
        d_ff: int,
        heads_num: int,
        dropout_rate: float,
        layer_norm_eps: float,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.embedding = Embedding(tgt_vocab_size, d_model, pad_idx)

        self.positional_encoding = AddPositionalEncoding(
            d_model,
            max_len,
            device,
        )

        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model,
                    d_ff,
                    heads_num,
                    dropout_rate,
                    layer_norm_eps,
                )
                for _ in range(N)
            ]
        )

    def forward(
        self,
        tgt: torch.Tensor,  # Output of multi-head attention
        src: torch.Tensor,  # Output of encoder
        mask_src_tgt: torch.Tensor,
        mask_self: torch.Tensor,
    ) -> torch.Tensor:
        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        for decoder_layers in self.decoder_layers:
            x = decoder_layers(
                tgt,
                src,
                mask_src_tgt,
                mask_self,
            )
        return x
