#!/usr/bin/env python3
from chess import Board, Move, Piece
from chessml.encoding.schema import PIECES_TO_CODES, BOARD_DIM, CASTLING_SLICE, ENPASSANT_IDX, TURN_IDX
from pathlib import Path
import sys, argparse
import chess.pgn
import numpy as np
import numpy.typing as npt


def encode_board(board: Board, move: Move) -> npt.NDArray[np.int8]:
  encoded_board: npt.NDArray[np.int8] = np.zeros(BOARD_DIM, dtype=np.int8)

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
  
  # Then enpassant
  encoded_board[ENPASSANT_IDX] = board.ep_square + 1 if board.ep_square is not None else 0

  return encoded_board
  
  
     
   


def build_cache(pgn_path: Path, n_games: int):
   game_counter: int = 0;
   with open(pgn_path) as pgn:
     while game_counter < n_games:
        # We save the old amount of bytes read, so we can go back to that point in file if needed
        old_cursor_pos = pgn.tell();

        # Read ONLY the headers instead of the whole game, so we don't do potentially unnecessary parsing 
        headers = chess.pgn.read_headers(pgn)
        if headers is None:
           break

        # Filter out the games that we don't need
        event_header = headers.get("Event", "")
        if headers.get("Termination") != "Normal" or not event_header.startswith("Rated") or "Classical" not in event_header:
           continue
        
        # Now, if we got here, it means that the game is appropriate for us.
        # However, our cursor already went past it in order to read its headers.
        # Thats why we go back to the old cursor position and parse the whole game.
        pgn.seek(old_cursor_pos)

        white_elo = int(headers["WhiteElo"])
        black_elo = int(headers["BlackElo"])
        
        game = chess.pgn.read_game(pgn)
        if game is None:
           print(f"Error: game is None after read_game call. ")
           break

        board = game.board()


        move_cap = n_games * 150 # Upper limit for amount of moves we hold
        positions = np.empty((move_cap, BOARD_DIM), dtype=np.int8)

        print("Going through individual moves for current game...")
        for i, move in enumerate(game.mainline_moves()):
           
           positions[i] = encode_board(board, move)
           
           # Play the actual move
           board.push(move)

        print("Finished with all moves for current game.")

           
        print("Game counter: ", game_counter)
        
        
        game_counter = game_counter + 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pgn", help="Path to the PGN file", type=Path, required=True)
    parser.add_argument("-s", "--split", help="Specifies which split is the pgn file used for", required=True, choices=["trainval", "test"])
    parser.add_argument("-g", "--games", help="Specifies how many games will be used for the split", type=int, required=True)
    args = parser.parse_args()

    if not args.pgn.is_file():
        parser.error(f"Hello cruel world, {args.pgn} is not a file!")

    build_cache(pgn_path=args.pgn, n_games=args.games)
   

# TODO:
# 1) Move the helper functions if needed
# 2) [Important] Check whether the headers + tell/seek is actually helping or not
# 3) Apparently Lichess changed the naming of some modes at some point. Maybe we should rely on TimeControl header instead of the naming?
# 4) Color flipping?
