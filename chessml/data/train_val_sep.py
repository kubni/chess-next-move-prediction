#!/usr/bin/env python3
from chessml.encoding.schema import DEFAULT_CACHE_DIR, CACHE_FILES, META_GAME_ID, META_DTYPE

import numpy as np
import numpy.typing as npt
import argparse
from pathlib import Path


# NOTE: Apparently numpy arrays don't like things like "not" and use "~" instead.

def save_train_val_ids_separately(out_dir: Path = DEFAULT_CACHE_DIR, val_every_nth: int = 20) -> None:
    meta = np.load(out_dir / CACHE_FILES["meta"].format(split="trainval"))
    game_ids: npt.NDArray[META_DTYPE]  = meta[:, META_GAME_ID] # Get the whole GAME_ID column from the meta matrix

    n_games = game_ids.max() + 1 
    all_game_ids = np.arange(n_games)

    is_val_mask = all_game_ids % val_every_nth == 0
    val_game_ids = all_game_ids[is_val_mask]
    train_game_ids = all_game_ids[~is_val_mask]

    # Assert:
    # 1) that we don't have any intersections between training and val sets:
    assert not (set(train_game_ids) & set(val_game_ids))

    # 2) That we didn't skip anything
    assert len(train_game_ids) + len(val_game_ids) == n_games

    # 3) That we properly built validation set
    assert len(val_game_ids) > 0

    # Cache the separated train and val indices, which will be used to differentiate which position
    np.save(file=out_dir / CACHE_FILES["train_game_ids"], arr=train_game_ids)
    np.save(file=out_dir / CACHE_FILES["val_game_ids"], arr=val_game_ids)



def games_positions_mask(meta: npt.NDArray[META_DTYPE], game_ids: npt.NDArray[META_DTYPE]) -> npt.NDArray[np.bool]:
    """
    Returns a boolean mask (an array of True/False) corresponding to the given game ids.
    For every game from game_id, we will get X True values in succession, where X is the number of moves in that game.
    This will allow us to filter positions and labels and get only the positions of games that we want to use.
    Example:
        mask = games_positions_mask(meta, val_game_ids)
        val_positions = positions[mask]
    """
    return np.isin(meta[:, META_GAME_ID], game_ids)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="Specifies the output directory where the generated cache files will be saved.", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("-n", "--val-every-nth", help="Every n-th game goes to the validation set, the rest goes to the training set. Value of n=20 means 5%%.", type=int, default=20)
    args = parser.parse_args()
    save_train_val_ids_separately(args.output, args.val_every_nth)
    




    
