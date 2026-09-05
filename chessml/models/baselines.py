#!/usr/bin/env python3
from chessml.encoding.schema import METRICS_DIR
from chessml.models.common_stuff import build_and_train, evaluate, as_predict_fn, get_necessary_stuff_for_training
from chessml.encoding.move_vocab import VOCAB_SIZE, load_move_vocab
from chessml.encoding.board_planes import FEATURES, to_planes
import torch.nn as nn
import json
def build_logreg() -> nn.Module:
    return nn.Sequential(nn.Flatten(),
                         nn.Linear(FEATURES, VOCAB_SIZE))

if __name__ == '__main__':
    print(f"Total params: {sum(p.numel() for p in build_logreg().parameters())}")

    data_for_training = get_necessary_stuff_for_training()

    positions = data_for_training["positions"]
    labels = data_for_training["labels"]
    meta = data_for_training["meta"]
    val_positions_indices = data_for_training["val_positions_indices"]

    model, lr_results = build_and_train(
        build_model_fn=build_logreg,
        model_name="logreg",
        batch_size=4096,
        num_epochs=40,
        data_for_training=data_for_training,
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "logreg_lr_results.json").write_text(
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

    (METRICS_DIR / "logreg_val_eval.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    
