import json
from pathlib import Path

import chess

from chessml.encoding.schema import VOCAB_PATH

VOCAB_SIZE = 1968

# Offsets as (column, row). The first four are rook-like, the last four bishop-like.
_QUEEN_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
_KNIGHT_DELTAS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]

# A queen promotion is written out in UCI (e7e8q), so it is a class of its own,
# distinct from e7e8.
_PROMOTION_PIECES = "nbrq"


def _name(column: int, row: int) -> str:
    """Board coordinates -> square name, e.g. (4, 3) -> 'e4'."""
    return chess.square_name(chess.square(column, row))


def build_move_vocab() -> list[str]:
    """
    Build every geometrically possible UCI move.

    Legality in a concrete position is irrelevant here - the only question is whether
    two squares can be connected by some piece.
    Returns:
        A sorted list of VOCAB_SIZE UCI strings.
    """
    moves: set[str] = set()

    for from_square in chess.SQUARES:
        column = chess.square_file(from_square)
        row = chess.square_rank(from_square)
        name = chess.square_name(from_square)

        # Queen rays also cover the rook, the bishop, the king and castling
        for d_column, d_row in _QUEEN_DIRS:
            for distance in range(1, 8):
                new_column = column + d_column * distance
                new_row = row + d_row * distance
                if not (0 <= new_column < 8 and 0 <= new_row < 8):
                    break  # off the board
                moves.add(name + _name(new_column, new_row))

        for d_column, d_row in _KNIGHT_DELTAS:
            new_column, new_row = column + d_column, row + d_row
            if 0 <= new_column < 8 and 0 <= new_row < 8:
                moves.add(name + _name(new_column, new_row))

    # Promotions: white from row 6 to 7, black from row 1 to 0,
    # either straight ahead or capturing diagonally.
    for from_row, to_row in ((6, 7), (1, 0)):
        for column in range(8):
            for new_column in (column - 1, column, column + 1):
                if not 0 <= new_column < 8:
                    continue
                base = _name(column, from_row) + _name(new_column, to_row)
                for piece in _PROMOTION_PIECES:
                    moves.add(base + piece)

    return sorted(moves)


def save_move_vocab(vocab: list[str], path: Path = VOCAB_PATH) -> None:
    """Write the vocabulary as a JSON list; position in the list is the class index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab), encoding="utf-8")


def load_move_vocab(path: Path = VOCAB_PATH) -> tuple[list[str], dict[str, int]]:
    """
    Load the vocabulary from disk.

    Returns:
        (index_to_move, move_to_index).

    Raises:
        FileNotFoundError: If the vocabulary has not been generated yet; the message
            names the command that generates it.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"no vocabulary at {path}; run: python -m chessml.encoding.move_vocab"
        )
    index_to_move: list[str] = json.loads(path.read_text(encoding="utf-8"))
    return index_to_move, {uci: i for i, uci in enumerate(index_to_move)}


if __name__ == "__main__":
    vocabulary = build_move_vocab()
    save_move_vocab(vocabulary)
    print(f"{len(vocabulary)} moves -> {VOCAB_PATH}")