import json

import numpy as np
import torch
import torch.nn as nn

from chessml.encoding.move_vocab import VOCAB_SIZE, load_move_vocab
from chessml.encoding.schema import (
    CACHE_FILES, DEFAULT_CACHE_DIR, METRICS_DIR, META_GAME_ID, META_PLY,
)
from chessml.encoding.tokenizer import CONTEXT_LEN, PAD_ID, TOKEN_VOCAB_SIZE, build_contexts
from chessml.eval.metrics import evaluate
from chessml.models.common_stuff import get_necessary_stuff_for_training, pick_device
from chessml.train import RANDOM_STATE, train_model


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

        # one position -> one embedding, because attention is blind to order:
        # without this, "e4 e5 nf3" == "e5 e4 nf3"
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




LEARNING_RATE = 3e-4

BATCH_SIZE = 512
NUM_EPOCHS = 30

def forward_fn(model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    return model(tokens)

def as_token_predict_fn(model: nn.Module, device: str | None = None):
    device = device or pick_device()
    model = model.to(device).eval()

    @torch.no_grad()
    def predict(tokens_batch: np.ndarray) -> torch.Tensor:
        return model(torch.from_numpy(tokens_batch).to(device)).float().cpu()

    return predict

def check_cache_contract(
    seq_flat: np.ndarray,
    seq_off: np.ndarray,
    labels: np.ndarray,
    meta: np.ndarray,
    sample: int = 1000,
) -> None:
    """
    Assert the contract seq_flat[seq_off[g] + ply] == labels[i]
    """
    rows = np.random.default_rng(RANDOM_STATE).choice(
        len(labels), min(sample, len(labels)), replace=False
    )
    g, ply = meta[rows, META_GAME_ID], meta[rows, META_PLY]
    broken = int((seq_flat[seq_off[g] + ply] != labels[rows]).sum())
    if broken:
        raise ValueError(
            f"cache contract broken on {broken}/{len(rows)} sampled rows: "
            "meta and seq_flat disagree, so token windows would be wrong"
        )


if __name__ == '__main__':
    data_for_training = get_necessary_stuff_for_training()

    positions = data_for_training["positions"]
    labels = data_for_training["labels"]
    meta = data_for_training["meta"]
    train_positions_indices = data_for_training["train_positions_indices"]
    val_positions_indices = data_for_training["val_positions_indices"]

    seq_flat = np.load(DEFAULT_CACHE_DIR / CACHE_FILES["seq_flat"].format(split="trainval"))
    seq_off = np.load(DEFAULT_CACHE_DIR / CACHE_FILES["seq_off"].format(split="trainval"))
    check_cache_contract(seq_flat, seq_off, labels, meta)

    # One window per cached position, so a row of tokens and a row of positions
    # describe the same moment and train_model can index either with the same split.
    tokens = build_contexts(seq_flat, meta, seq_off)

    torch.manual_seed(RANDOM_STATE)
    model = MoveTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    metrics, best_val_accuracy = train_model(
        model=model,
        forward_fn=forward_fn,
        optimizer=optimizer,
        number_of_epochs=NUM_EPOCHS,
        train_positions_indices=train_positions_indices,
        val_positions_indices=val_positions_indices,
        positions=tokens,
        labels=labels,
        name="transformer",
        config=model.config, # notebook can do MoveTransformer(**checkpoint["config"])  without retyping hyperparameters.
        batch_size=BATCH_SIZE,
        seed=RANDOM_STATE,
    )

    _, move_to_index = load_move_vocab()
    result = evaluate(
        as_token_predict_fn(model),
        positions[val_positions_indices],
        labels[val_positions_indices],
        meta[val_positions_indices],
        move_to_index,
        model_inputs=tokens[val_positions_indices],
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "transformer_config.json").write_text(
        json.dumps(
            {
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "max_epochs": NUM_EPOCHS,
                "seed": RANDOM_STATE,
                "best_val_accuracy": best_val_accuracy,
                "architecture": model.config,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (METRICS_DIR / "transformer_val_evaluate.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )