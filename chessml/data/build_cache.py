#!/usr/bin/env python3
from chessml.encoding.move_vocab import load_move_vocab
from typing import Literal
from chess import Board, Move, Piece
from chessml.encoding.schema import PIECES_TO_CODES, BOARD_DIM, CASTLING_SLICE, ENPASSANT_IDX, TURN_IDX, CACHE_FILES, DEFAULT_CACHE_DIR, META_DIM, elo_bucket, META_DTYPE, LABEL_DTYPE, POSITION_DTYPE, SEQ_DTYPE, OFF_DTYPE
from pathlib import Path
import argparse
import chess.pgn
import numpy as np
import numpy.typing as npt


def encode_board(board: Board) -> npt.NDArray[np.int8]:
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
  
  
def build_cache(pgn_path: Path, n_games: int, vocab: dict[str, int], split: Literal["trainval", "test"], out_dir: Path = DEFAULT_CACHE_DIR):
   game_counter: int = 0;
   move_cap = min(n_games * 150, 50000000) # Upper limit for amount of moves we hold
   positions = np.empty((move_cap, BOARD_DIM), dtype=POSITION_DTYPE)
   labels = np.empty(move_cap, dtype=LABEL_DTYPE)
   meta = np.empty((move_cap, META_DIM), dtype=META_DTYPE) # TODO: np.int16?
   seq_flat = np.empty(move_cap, dtype=SEQ_DTYPE)
   seq_off = np.empty(n_games + 1, dtype=OFF_DTYPE)
   seq_off[0] = 0
   
   i = 0;
   with open(pgn_path) as pgn:
     while game_counter < n_games:
        # We save the old amount of bytes read, so we can go back to that point in file if needed
        old_cursor_pos = pgn.tell();

        # Read ONLY the headers instead of the whole game, so we don't do potentially unnecessary parsing 
        headers = chess.pgn.read_headers(pgn)
        if headers is None:
          # End of the file. We report how many games we actually parsed, since it can be lower than n_games.
          print(f"The number of actually parsed games: {game_counter}.\n The number of games we wanted to parse: {n_games}.")
          break

        # Filter out the games that we don't need
        event_header = headers.get("Event", "")
        if headers.get("Termination") != "Normal" or not event_header.startswith("Rated") or "Classical" not in event_header:
           continue
        
        # Sometimes the elo is unknown (denoted by a '?' mark),
        # so we skip such games
        white_elo = headers.get("WhiteElo", "")
        black_elo = headers.get("BlackElo", "")
        if not (white_elo.isdecimal() and black_elo.isdecimal()):
          continue

        white_elo = int(white_elo)
        black_elo = int(black_elo)
         
        # If we got here, it means that the game is appropriate.
        # However, our cursor already went past it in order to read its headers.
        # Thats why we go back to the old cursor position and parse the whole game.
        pgn.seek(old_cursor_pos)

        
        game = chess.pgn.read_game(pgn)
        if game is None:
           print(f"Error: game is None after read_game call. ")
           break

        board = game.board()
        for move in game.mainline_moves():
           # Flip the board when black is playing
           if board.turn == chess.BLACK:
              view = board.mirror()
              mv = chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), move.promotion)
           else:
              view, mv = board, move

           positions[i] = encode_board(view)
           encoded_move = vocab[mv.uci()]
           labels[i] = encoded_move 
           meta[i] = [game_counter, board.ply(), elo_bucket(white_elo if board.turn == chess.WHITE else black_elo)]
           seq_flat[i] = encoded_move
           
           i += 1
           # Play the actual move
           board.push(move)

        seq_off[game_counter + 1] = i  
        game_counter += 1

   # Cache the positions
   out_dir.mkdir(parents=True, exist_ok=True)
   np.save(file=out_dir / CACHE_FILES["positions"].format(split=split), arr=positions[:i])  # :i because we probably have some empty ones left over (due to generous move_cap)
   np.save(file=out_dir / CACHE_FILES["labels"].format(split=split), arr=labels[:i])  
   np.save(file=out_dir / CACHE_FILES["meta"].format(split=split), arr=meta[:i])  
   np.save(file=out_dir / CACHE_FILES["seq_flat"].format(split=split), arr=seq_flat[:i])  
   np.save(file=out_dir / CACHE_FILES["seq_off"].format(split=split), arr=seq_off[:game_counter + 1])  
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pgn", help="Path to the PGN file", type=Path, required=True)
    parser.add_argument("-s", "--split", help="Specifies which split is the pgn file used for", required=True, choices=["trainval", "test"])
    parser.add_argument("-g", "--games", help="Specifies how many games will be used for the split", type=int, required=True)
    parser.add_argument("-o", "--output", help="Specifies the output directory where the generated cache files will be saved.", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()

    if not args.pgn.is_file():
        parser.error(f"Hello cruel world, {args.pgn} is not a file!")

    _, vocab = load_move_vocab()
    build_cache(pgn_path=args.pgn, n_games=args.games, vocab=vocab, split=args.split, out_dir=args.output)
   

# TODO:
# 1) Move the helper functions if needed
# 2) [Important] Check whether the headers + tell/seek is actually helping or not
# 3) Apparently Lichess changed the naming of some modes at some point. Maybe we should rely on TimeControl header instead of the naming?
# 4) Color flipping/mirroring?  

# 5) [Important] logging instead of prints 
