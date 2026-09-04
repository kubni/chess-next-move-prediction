#!/usr/bin/env python3
from chessml.encoding.move_vocab import VOCAB_SIZE
from chessml.encoding.schema import POSITION_DTYPE, LABEL_DTYPE, CACHE_FILES, DEFAULT_CACHE_DIR

from typing import Iterator

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
                              positions: npt.NDArray[POSITION_DTYPE],
                              labels: npt.NDArray[LABEL_DTYPE],
                              batch_size: int,
                              num_epochs: int,
                              lr: float
):
    device = pick_device()
    model = build_logistic_regression().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for start in range(0, len(train_positions_indices), batch_size):
            idx = train_positions_indices[start:start + batch_size]
            x = torch.from_numpy(positions[idx]).to(device)
            y = torch.from_numpy(labels[idx].astype(np.int64)).to(device)

            loss = F.cross_entropy(model(to_planes(x)), y)

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(epoch, loss.item())    






d = DEFAULT_CACHE_DIR
positions = np.load(d / CACHE_FILES["positions"].format(split="trainval"))
labels    = np.load(d / CACHE_FILES["labels"].format(split="trainval"))
meta      = np.load(d / CACHE_FILES["meta"].format(split="trainval"))
train_game_ids = np.load(d / CACHE_FILES["train_game_ids"])

train_positions_indices = np.flatnonzero(games_positions_mask(meta, train_game_ids))

train_logistic_regression(train_positions_indices, positions, labels, batch_size=4096, num_epochs=10, lr=1e-3)
 

