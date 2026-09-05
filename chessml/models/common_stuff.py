#!/usr/bin/env python3
from collections.abc import Callable
from chessml.eval.metrics import evaluate
from chessml.data.train_val_sep import games_positions_mask
from chessml.encoding.schema import POSITION_DTYPE, LABEL_DTYPE, DEFAULT_CACHE_DIR, CACHE_FILES
from chessml.train import train_model, RANDOM_STATE
from chessml.encoding.move_vocab import VOCAB_SIZE, load_move_vocab
from chessml.encoding.board_planes import FEATURES, to_planes

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import numpy.typing as npt



def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


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
    build_model_fn: Callable[[], nn.Module],
    train_positions_indices: npt.NDArray[np.int64],
    val_positions_indices: npt.NDArray[np.int64],
    positions: npt.NDArray[POSITION_DTYPE],
    labels: npt.NDArray[LABEL_DTYPE],
    name: str,
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

        model = build_model_fn()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        metrics, val_acc = train_model(model=model,
                    forward_fn=forward_fn,
                    optimizer=optimizer,
                    number_of_epochs=num_epochs,
                    train_positions_indices=train_positions_indices,
                    val_positions_indices=val_positions_indices,
                    positions=positions,
                    labels=labels,
                    name=name,
                    config={"lr": lr},
                    batch_size=batch_size,
                    seed=RANDOM_STATE
                    )

        lr_results[str(lr)] = val_acc

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model

    if best_model is None:
        raise ValueError("Error: learning_rates must not be empty")
    
    return lr_results, best_model



def get_necessary_stuff_for_training() -> dict:
    d = DEFAULT_CACHE_DIR
    positions = np.load(d / CACHE_FILES["positions"].format(split="trainval"))
    labels    = np.load(d / CACHE_FILES["labels"].format(split="trainval"))
    meta      = np.load(d / CACHE_FILES["meta"].format(split="trainval"))
    train_game_ids = np.load(d / CACHE_FILES["train_game_ids"])
    val_game_ids = np.load(d / CACHE_FILES["val_game_ids"])

    train_positions_indices = np.flatnonzero(games_positions_mask(meta, train_game_ids))
    val_positions_indices = np.flatnonzero(games_positions_mask(meta, val_game_ids))
  

    return {
        "positions": positions,
        "labels": labels,
        "meta": meta,
        "train_positions_indices": train_positions_indices,
        "val_positions_indices": val_positions_indices,
    }



def build_and_train(build_model_fn: Callable[[], nn.Module], model_name: str, batch_size: int, num_epochs: int, data_for_training: dict) -> tuple[nn.Module, dict[str, float]]:

    positions = data_for_training["positions"]
    labels = data_for_training["labels"]
    meta = data_for_training["meta"]
    train_positions_indices = data_for_training["train_positions_indices"]
    val_positions_indices = data_for_training["val_positions_indices"]

    # Test multiple learning rates and find the best model
    lr_results, model = try_multiple_learning_rates(
        build_model_fn,
        train_positions_indices,
        val_positions_indices,
        positions,
        labels,
        model_name,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )

    return model, lr_results
