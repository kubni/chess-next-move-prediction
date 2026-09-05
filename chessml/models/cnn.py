#!/usr/bin/env python3
from chessml.train import RANDOM_STATE
from chessml.encoding.schema import METRICS_DIR
from chessml.models.common_stuff import (
    build_and_train, evaluate, as_predict_fn, get_necessary_stuff_for_training,
)
from chessml.encoding.move_vocab import VOCAB_SIZE, load_move_vocab
from chessml.encoding.board_planes import N_PLANES

import torch
import torch.nn as nn
import torch.nn.functional as F

import json

class ChessConvClassifier(nn.Module):
    def __init__(self, number_of_classes):
        super(ChessConvClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=N_PLANES, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        self.gn1 = nn.GroupNorm(8, 64)
        self.gn2 = nn.GroupNorm(8, 128)
        self.gn3 = nn.GroupNorm(8, 128)
        self.gn4 = nn.GroupNorm(8, 128)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(in_features=128 * 64, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=number_of_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.gn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.gn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.gn4(x)
        x = F.relu(x)
        
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        logits = self.fc2(x)

        return logits


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    for name, layer in model.named_children():
        num_params = sum(p.numel() for p in layer.parameters())
        print(f"Layer: {name}, Parameters: {num_params}")
    return total_params


def build_cnn() -> nn.Module:
    return ChessConvClassifier(number_of_classes=VOCAB_SIZE)


if __name__ == '__main__':
    print(f"Total params: {count_parameters(build_cnn())}")

    data_for_training = get_necessary_stuff_for_training()

    positions = data_for_training["positions"]
    labels = data_for_training["labels"]
    meta = data_for_training["meta"]
    val_positions_indices = data_for_training["val_positions_indices"]

    model, lr_results = build_and_train(
        build_model_fn=build_cnn,
        model_name="cnn",
        batch_size=2048,
        num_epochs=40,
        data_for_training=data_for_training,
        learning_rates=(1e-3,),
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "cnn_lr_results.json").write_text(
        json.dumps(lr_results, indent=2), encoding="utf-8"
    )

    _, move_to_index = load_move_vocab()
    result = evaluate(
        as_predict_fn(model),
        positions[val_positions_indices],
        labels[val_positions_indices],
        meta[val_positions_indices],
        move_to_index,
    )

    (METRICS_DIR / "cnn_eval.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
