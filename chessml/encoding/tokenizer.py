import numpy as np
import numpy.typing as npt

from chessml.encoding.move_vocab import VOCAB_SIZE
from chessml.encoding.schema import META_GAME_ID

PAD_ID = 0 #  Used to fill out the history with 0
BOS_ID = 1 # Used so softmax does not do 0/0 if all tokens are PAD
MOVE_OFFSET = 2
TOKEN_VOCAB_SIZE = VOCAB_SIZE + MOVE_OFFSET

CONTEXT_LEN = 64

TOKEN_DTYPE = np.int16   # ids reach 1969, same range argument as LABEL_DTYPE


def build_contexts(
    labels: npt.NDArray[np.int16],
    meta: npt.NDArray[np.int32],
    seq_off: npt.NDArray[np.int32],
    context_len: int = CONTEXT_LEN,
) -> npt.NDArray[np.int16]:
    """
    Build one token window per cached position.

    Row i of the result is the history the transformer sees when it has to predict
    labels[i]: the moves of that game played before row i, most recent one last.

    Returns:
        (N, context_len) token ids.
    """
    n = len(labels)
    tokens = np.full((n, context_len), PAD_ID, dtype=TOKEN_DTYPE)

    for i in range(n):
        start = int(seq_off[meta[i, META_GAME_ID]])

        # Window starts at start only if start of the game should be inside
        # the window.
        first = max(start, i - context_len)
        window = list(labels[first:i] + MOVE_OFFSET)

        # BOS marks the game start, and only appears when the start is still
        # inside the window.
        if start > i - context_len:
            window = [BOS_ID] + window

        # Left-padded: if window < context_len, the start of tokens will have PAD 
        # repeated. Most recent move one the last.
        tokens[i, context_len - len(window):] = window

    return tokens