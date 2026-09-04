import torch
import torch.nn.functional as F

from chessml.encoding.schema import EXTRAS_SLICE, SQUARES_SLICE

N_PIECE_PLANES = 12         
N_EXTRA_PLANES = 5           
N_PLANES = N_PIECE_PLANES + N_EXTRA_PLANES
FEATURES = N_PLANES * 64     # 1088 — flat input width for logistic regression and MLP


def to_planes(boards: torch.Tensor) -> torch.Tensor:
    """
    Encode a batch of cache rows as stacked 8x8 planes.

    Returns:
        (B, N_PLANES, 8, 8).
    """
    batch = boards.shape[0]

    # Piece codes are labels, not magnitudes, so they cannot be fed in raw: a linear
    # layer would read code 12 as "twelve times" code 1, an ordering that does not
    # exist. Category 0 (empty) is dropped — it is implied by the other twelve.
    squares = boards[:, SQUARES_SLICE].long()
    onehot = F.one_hot(squares, num_classes=N_PIECE_PLANES + 1)      # (B, 64, 13)

    pieces = onehot[:, :, 1:]                    # (B, 64, 12)
    # conv2d expects (B,C,H,W)
    pieces = pieces.permute(0, 2, 1)             # (B, 12, 64)
    pieces = pieces.reshape(batch, N_PIECE_PLANES, 8, 8).float()

    # Castling rights and side to move are not tied to a square
    extras = boards[:, EXTRAS_SLICE].float()                # (B, 5)
    extras = extras.reshape(batch, N_EXTRA_PLANES, 1, 1)    # one scalar per plane
    extras = extras.expand(batch, N_EXTRA_PLANES, 8, 8)     # same value on every square

    return torch.cat([pieces, extras], dim=1)