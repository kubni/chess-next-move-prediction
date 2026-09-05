#!/usr/bin/env python3
from chessml.train import RANDOM_STATE, train_model
from chessml.eval.metrics import evaluate
import json
from chessml.encoding.move_vocab import VOCAB_SIZE, load_move_vocab
from chessml.encoding.schema import POSITION_DTYPE, LABEL_DTYPE, CACHE_FILES, DEFAULT_CACHE_DIR, MODELS_DIR, METRICS_DIR

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F

from chessml.encoding.board_planes import FEATURES, to_planes
from chessml.data.train_val_sep import games_positions_mask

def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def build_logistic_regression() -> nn.Module:
    return nn.Sequential(nn.Flatten(),
                         nn.Linear(FEATURES, VOCAB_SIZE))


def forward_fn(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model(to_planes(x))
   
def as_predict_fn(model: nn.Module, device: str | None = None):
    """
    Wraps given model into the format that evaluate() wants.
    """
    device = device or pick_device()
    model = model.to(device).eval()

    @torch.no_grad()
    def predict(positions_batch: npt.NDArray[POSITION_DTYPE]) -> torch.Tensor:
        x = torch.from_numpy(positions_batch).to(device)
        return model(to_planes(x)).float().cpu()

    return predict



def try_multiple_learning_rates(
    train_positions_indices: npt.NDArray[np.int64],
    val_positions_indices: npt.NDArray[np.int64],
    positions: npt.NDArray[POSITION_DTYPE],
    labels: npt.NDArray[LABEL_DTYPE],
    learning_rates: tuple[float, ...] = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
    batch_size: int = 4096,
    num_epochs: int = 150,
) -> tuple[dict[str, float], nn.Module]:

    torch.manual_seed(RANDOM_STATE)
    lr_results = {}
    best_model: nn.Module | None = None
    best_acc = -1.0
    for lr in learning_rates:
        # print("LR: ", lr, "\n ACC: ", acc)

        model = build_logistic_regression()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        metrics = train_model(model=model,
                    forward_fn=forward_fn,
                    optimizer=optimizer,
                    number_of_epochs=num_epochs,
                    train_positions_indices=train_positions_indices,
                    val_positions_indices=val_positions_indices,
                    positions=positions,
                    labels=labels,
                    name="logreg",
                    config={},
                    batch_size=batch_size,
                    seed=RANDOM_STATE
                    )

        acc = max(metrics["val_accuracy"])
        lr_results[str(lr)] = acc

        if acc > best_acc:
            best_acc = acc
            best_model = model

    if best_model is None:
        raise ValueError("Error: learning_rates must not be empty")
    
    return lr_results, best_model

    





if __name__ == '__main__':
    d = DEFAULT_CACHE_DIR
    positions = np.load(d / CACHE_FILES["positions"].format(split="trainval"))
    labels    = np.load(d / CACHE_FILES["labels"].format(split="trainval"))
    meta      = np.load(d / CACHE_FILES["meta"].format(split="trainval"))
    train_game_ids = np.load(d / CACHE_FILES["train_game_ids"])
    val_game_ids = np.load(d / CACHE_FILES["val_game_ids"])

    train_positions_indices = np.flatnonzero(games_positions_mask(meta, train_game_ids))
    val_positions_indices = np.flatnonzero(games_positions_mask(meta, val_game_ids))
  
    batch_size = 4096
    num_epochs = 150
    
    # Test multiple learning rates and find the best model
    lr_results, model = try_multiple_learning_rates(
        train_positions_indices,
        val_positions_indices,
        positions,
        labels,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )

    best_lr = float(max(lr_results, key=lambda k: lr_results[k]))

    _, move_to_index = load_move_vocab()

    result = evaluate(
        as_predict_fn(model),
        positions[val_positions_indices],
        labels[val_positions_indices],
        meta[val_positions_indices],
        move_to_index,
    )
    assert result["top1"] == lr_results[str(best_lr)]
        
    print("Best lr: ", best_lr, "\n")
    print("Result: ", result)
