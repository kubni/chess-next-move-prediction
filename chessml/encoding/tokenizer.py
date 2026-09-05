import numpy as np
import numpy.typing as npt

from chessml.encoding.move_vocab import VOCAB_SIZE
from chessml.encoding.schema import META_GAME_ID, META_PLY

PAD_ID = 0 #  Used to fill out the history with 0
BOS_ID = 1 # Used so softmax does not do 0/0 if all tokens are PAD
MOVE_OFFSET = 2
TOKEN_VOCAB_SIZE = VOCAB_SIZE + MOVE_OFFSET

CONTEXT_LEN = 64

TOKEN_DTYPE = np.int16   # ids reach 1969, same range argument as LABEL_DTYPE


def build_contexts(
    seq_flat: npt.NDArray[np.int16],
    meta: npt.NDArray[np.int32],
    seq_off: npt.NDArray[np.int32],
    context_len: int = CONTEXT_LEN,
) -> npt.NDArray[np.int16]:
    """
     Build one token window per cached position.

    Row i of the result is the history the model sees when it has to predict
    labels[i]: the moves of that game played before that position, most recent last.

    Args:
        seq_flat: every move of every game in this split, concatenated.
        meta: (N, META_DIM), aligned row for row with the positions to tokenize.
        seq_off: (G + 1,) where each game starts inside seq_flat.

    Returns:
        (N, context_len) left-padded token ids.
    """
    n = len(meta)
    tokens = np.full((n, context_len), PAD_ID, dtype=TOKEN_DTYPE)

    for i in range(n):
        start = int(seq_off[meta[i, META_GAME_ID]])
        ply = int(meta[i, META_PLY])

        # The moves played before this position, cut down to the last context_len.
        first = start + max(0, ply - context_len)
        window = list(seq_flat[first:start + ply] + MOVE_OFFSET)


        # BOS marks the game start and only fits while that start is still inside
        # the window. It also guarantees one non-pad token in every row, so no
        # row is entirely masked out of attention.
        if ply < context_len:
            window = [BOS_ID] + window

        # Left-padded: if window < context_len, the start of tokens will have PAD 
        # repeated. Most recent move one the last.
        tokens[i, context_len - len(window):] = window

    return tokens