import numpy as np
import numpy.typing as npt
import torch

from chessml.encoding.board_codec import decode_board
from chessml.encoding.move_vocab import VOCAB_SIZE


def legal_mask(
    positions: npt.NDArray[np.int8],
    move_to_index: dict[str, int],
) -> npt.NDArray[np.bool]:
    """
    Build the legal-move mask for a batch of cached positions.

    Returns:
        (B, VOCAB_SIZE) boolean array, True where the move is legal.
    """
    mask = np.zeros((len(positions), VOCAB_SIZE), dtype=np.bool)

    for i in range(len(positions)):
        board = decode_board(positions[i])
        for move in board.legal_moves:
            mask[i, move_to_index[move.uci()]] = True

    return mask


def apply_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Make illegal moves impossible.

    Args:
        logits: (B, VOCAB_SIZE) raw model output.
        mask: (B, VOCAB_SIZE) boolean, True where legal.

    Returns:
        A copy of `logits` with -inf where move is illegal 
        (thats why we do ~mask), so they cant win softmax.
    """
    return logits.masked_fill(~mask, -float("inf"))