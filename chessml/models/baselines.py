#!/usr/bin/env python3
import json
from chessml.encoding.move_vocab import VOCAB_SIZE
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


def train_logistic_regression(train_positions_indices: npt.NDArray[np.int64],
                              val_positions_indices: npt.NDArray[np.int64],
                              positions: npt.NDArray[POSITION_DTYPE],
                              labels: npt.NDArray[LABEL_DTYPE],
                              batch_size: int,
                              num_epochs: int,
                              lr: float
) -> tuple[nn.Module, float]:
    # Set a seed that torch.randperm and starting weights will use
    torch.manual_seed(0)
    
    device = pick_device()
    model = build_logistic_regression().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Move data to gpu
    positions_gpu = torch.from_numpy(positions).to(device)
    labels_gpu = torch.from_numpy(labels.astype(np.int64)).to(device)
    train_indices_gpu = torch.from_numpy(train_positions_indices).to(device)
    val_indices_gpu = torch.from_numpy(val_positions_indices).to(device)
    
    best_val_acc = -1.0;
    best_model_state = None
    patience = 2
    bad_epochs = 0
    min_delta = 1e-4
    
    for epoch in range(num_epochs):
        train_indices_gpu_permuted = train_indices_gpu[torch.randperm(len(train_indices_gpu), device=device)]
        
        for start in range(0, len(train_indices_gpu_permuted), batch_size):
            idx = train_indices_gpu_permuted[start:start + batch_size]
            x = positions_gpu[idx]
            y = labels_gpu[idx]

            loss = F.cross_entropy(model(to_planes(x)), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Calculate accuracy on validation set and hopefully detect overfitting
        with torch.no_grad():
            correct = 0
            for start in range(0, len(val_positions_indices), batch_size):
                idx = val_indices_gpu[start:start + batch_size]
                x = positions_gpu[idx]
                y = labels_gpu[idx]
                correct += (model(to_planes(x)).argmax(1) == y).sum().item()

        val_acc_current = correct / len(val_positions_indices)
        print(epoch, loss.item(), correct / len(val_positions_indices))


        if val_acc_current > best_val_acc + min_delta:
            best_val_acc = val_acc_current

            # Save the model state that is best currently
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

            # Reset bad epoch counter
            bad_epochs = 0
        else:
            # If our current val accuracy is worse than the best, we don't break immediately,
            # but rather if it happens more than ``patience`` times.
            bad_epochs += 1
            if bad_epochs >= patience:
                break

        

    # We load the best found model state,
    # since we don't want to return the model state from when we "lost patience".
    if best_model_state is None:
        raise ValueError("[Error] num_epochs must be at least 1!")

    model.load_state_dict(best_model_state)
    return model, best_val_acc


   
def try_multiple_learning_rates(
    train_positions_indices: npt.NDArray[np.int64],
    val_positions_indices: npt.NDArray[np.int64],
    positions: npt.NDArray[POSITION_DTYPE],
    labels: npt.NDArray[LABEL_DTYPE],
    learning_rates: tuple[float, ...] = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
    batch_size: int = 4096,
    num_epochs: int = 150,
) -> tuple[dict[str, float], nn.Module]:

    lr_results = {}
    best_model: nn.Module | None = None
    best_acc = -1.0
    for lr in learning_rates:
        model, acc = train_logistic_regression(train_positions_indices, val_positions_indices, positions, labels, batch_size=batch_size, lr=lr, num_epochs=num_epochs)
        # print("LR: ", lr, "\n ACC: ", acc)
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

    # Save the results:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True) 
    (METRICS_DIR / "logreg_lr_testing.json").write_text(
        json.dumps(
            {
                "batch_size": batch_size,
                "max_epochs": num_epochs,
                "seed": 0,
                "results": lr_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Save the actual best found model
    torch.save(
        {
            "state_dict": {k: v.half() for k, v in model.state_dict().items()},
            "architecture": "logistic_regression",
            "vocab_size": VOCAB_SIZE,
            "features": FEATURES,
            "learning_rate": best_lr,
            "val_accuracy": lr_results[str(best_lr)],
        },
        MODELS_DIR / "logreg.pt",
    )


# TODO:
# 1) Utilize evaluate() and test set
