#!/usr/bin/env python3
from chessml.encoding.schema import POSITION_DTYPE, BOARD_DIM, CASTLING_SLICE, PIECES_TO_CODES, CODES_TO_PIECES, TURN_IDX, ENPASSANT_IDX, SQUARES_SLICE

import chess
from chess import Board, Piece
import numpy as np
import numpy.typing as npt


_CASTLING_ROOKS = (chess.BB_H1, chess.BB_A1, chess.BB_H8, chess.BB_A8)

def encode_board(board: Board) -> npt.NDArray[POSITION_DTYPE]:
  encoded_board: npt.NDArray[POSITION_DTYPE] = np.zeros(BOARD_DIM, dtype=np.int8)

  # First we get the piece codes
  piece_map: dict[int, Piece] = board.piece_map()
  for i, piece in piece_map.items():
     encoded_board[i] = PIECES_TO_CODES[(piece.piece_type, piece.color)]

  # Then the castling rights 
  encoded_board[CASTLING_SLICE] = [
     board.has_kingside_castling_rights(chess.WHITE),
     board.has_queenside_castling_rights(chess.WHITE),  
     board.has_kingside_castling_rights(chess.BLACK),
     board.has_queenside_castling_rights(chess.BLACK)  
  ]

  # Then whose turn it is
  encoded_board[TURN_IDX] = board.turn
  
  # Then enpassant. We increase by 1 in order to differentiate a1 and None.
  encoded_board[ENPASSANT_IDX] = board.ep_square + 1 if board.ep_square is not None else 0

  return encoded_board


def decode_board(position: npt.NDArray[POSITION_DTYPE]) -> Board:
    board = chess.Board(None)

    # We set only the attributes that we know
    # Whose turn it is (true = White)
    board.turn = bool(position[TURN_IDX]);

    for square_idx, code in enumerate(position[SQUARES_SLICE]):
        # Only decode if the square isn't empty
        if code:
            board.set_piece_at(square_idx, CODES_TO_PIECES[code])


    # Enpassant
    ep = int(position[ENPASSANT_IDX])
    board.ep_square = ep - 1 if ep else None


    # Castling rights
    rights = chess.BB_EMPTY
    for flag, rook_square in zip(position[CASTLING_SLICE], _CASTLING_ROOKS):
        if flag:
            rights |= rook_square
    board.castling_rights = rights

    return board;
