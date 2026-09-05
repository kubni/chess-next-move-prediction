#!/usr/bin/env python3
from chessml.models.common_stuff import build_and_train, evaluate, as_predict_fn, get_necessary_stuff_for_training
from chessml.encoding.move_vocab import VOCAB_SIZE, load_move_vocab
from chessml.encoding.board_planes import FEATURES, to_planes
import torch.nn as nn

def build_mlp():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(FEATURES, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, VOCAB_SIZE)
)

if __name__ == '__main__':
    data_for_training = get_necessary_stuff_for_training()
    
    positions = data_for_training["positions"]
    labels = data_for_training["labels"]
    meta = data_for_training["meta"]
    train_positions_indices = data_for_training["train_positions_indices"]
    val_positions_indices = data_for_training["val_positions_indices"]
    
    model, lr_results = build_and_train(build_model_fn=build_mlp, model_name="mlp", batch_size=4096, num_epochs=150, data_for_training=data_for_training)

    _, move_to_index = load_move_vocab()
    result = evaluate(
        as_predict_fn(model),
        positions[val_positions_indices],
        labels[val_positions_indices],
        meta[val_positions_indices],
        move_to_index,
    )

    # TODO: Use lr_results and result with matplotlib
