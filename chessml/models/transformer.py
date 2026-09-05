import torch
import torch.nn as nn

from chessml.encoding.move_vocab import VOCAB_SIZE
from chessml.encoding.tokenizer import CONTEXT_LEN, PAD_ID, TOKEN_VOCAB_SIZE


class MoveTransformer(nn.Module):
    """
    Predict the next move from the last context_len moves of the game.

    Args:
        d_model: width of the vector carried through every layer.
        nhead: attention heads; must divide d_model.
        num_layers: how many times the encoder block repeats.
        dim_feedforward: width of the MLP inside each block.
        dropout: applied to the embeddings and inside each block.
        context_len: window length, must match the tokenizer that built the input.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        context_len: int = CONTEXT_LEN,
    ) -> None:
        super().__init__()

        # Kept so save_checkpoint can record how to rebuild this model, instead of
        # relying on hyperparameters retyped in a notebook cell.
        self.config = {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "context_len": context_len,
        }

        # one token -> one embedding (row with 256 numbers) 
        self.token_embedding = nn.Embedding(TOKEN_VOCAB_SIZE, d_model, padding_idx=PAD_ID)

        # one position -> one embedding because attention is blind to position
        # ("e4 e5 nf3" == "nf3 e4 e5")
        self.position_embedding = nn.Embedding(context_len, d_model)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

        # VOCAB_SIZE, not TOKEN_VOCAB_SIZE: we don't need
        # to predict PAD and BOS from TOKEN_VOCAB.
        self.head = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, context_len) token ids from build_contexts.

        Returns:
            (B, VOCAB_SIZE) logits for the move played next.
        """
        tokens = tokens.long()   # the cache stores int16, nn.Embedding needs int64
        position_ids = torch.arange(tokens.size(1), device=tokens.device)

        x = self.dropout(self.token_embedding(tokens) + self.position_embedding(position_ids))

        # True marks positions attention must ignore. 
        # x = x + attention(norm(x))
        # x = x + MLP(norm(x))
        x = self.encoder(x, src_key_padding_mask=(tokens == PAD_ID))

        # This is not last move predicted, its last move + the summary of
        # game (like a conclusion of how every move influenced to get to this position).
        summary = x[:, -1]                  # (B, 256)

        summary = self.norm(summary)

        return self.head(summary)           # (B, VOCAB_SIZE)