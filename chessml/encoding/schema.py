from pathlib import Path
import chess
import numpy as np

# --- boards[i]: 70 numbers ---------------------------------------------------
BOARD_DIM = 70
SQUARES_SLICE = slice(0, 64)          # piece code per square, 0 = empty
CASTLING_SLICE = slice(64, 68)   # castling rights: K, Q, k, q (0/1)
TURN_IDX = 68                    # True = white to move
ENPASSANT_IDX = 69                      # 0 = no ep, otherwise square + 1 (1..64)
EXTRAS_SLICE = slice(CASTLING_SLICE.start, TURN_IDX + 1) #used for to planes encoding

# --- meta[i]: 3 numbers -------------------------------------------------------
META_DIM = 3
META_GAME_ID, META_PLY, META_ELO_BUCKET = 0, 1, 2

# META_GAME_ID indexes seq_off of the SAME split, not the game's position in the PGN.
# Contract: seq_flat[seq_off[meta[i, META_GAME_ID]] + meta[i, META_PLY]] == labels[i]

ELO_BUCKETS = [(0, 1200), (1200, 1500), (1500, 1800), (1800, 2100), (2100, 9999)]


def elo_bucket(elo: int) -> int:
    """Rating -> bucket index. Anything above the last bound falls into the last bucket."""
    for i, (low, high) in enumerate(ELO_BUCKETS):
        if low <= elo < high:
            return i
    return len(ELO_BUCKETS) - 1


# --- dtypes: chosen by value range --------------------------------------------
POSITION_DTYPE = np.int8    # 0..64
LABEL_DTYPE = np.int16   # move index 0..1967
META_DTYPE = np.int32    # game count exceeds int16
SEQ_DTYPE = np.int16     # same label space as labels
OFF_DTYPE = np.int32 

_pieces_and_colors =[(piece_type, color) for color in (chess.WHITE, chess.BLACK) for piece_type in range(chess.PAWN, chess.KING + 1)] 
PIECES_TO_CODES = {pc: i + 1 for i, pc in enumerate(_pieces_and_colors)}
CODES_TO_PIECES = {v: chess.Piece(*k) for k,v in PIECES_TO_CODES.items()}

CACHE_FILES = {
    "positions": "{split}_positions.npy",
    "train_game_ids": "train_game_ids.npy",
    "val_game_ids": "val_game_ids.npy",
    "labels":    "{split}_labels.npy",
    "meta":      "{split}_meta.npy",
    "seq_flat":  "{split}_seq_flat.npy",
    "seq_off":   "{split}_seq_off.npy",
    
}


PROJECT_ROOT = Path(__file__).resolve().parents[2]   
VOCAB_PATH = PROJECT_ROOT / "artifacts" / "move_vocab.json"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "artifacts" / "cache"


# TODO: Check whether some of this stuff should be moved to some other file
