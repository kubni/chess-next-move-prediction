#!/usr/bin/env python3
import sys, argparse
import chess.pgn
from pathlib import Path

def build_cache(pgn_path: Path):
   game_counter: int = 0;
   with open(pgn_path) as pgn:
     while game_counter < args.games:
        # We save the old amount of bytes read, so we can go back to that point in file if needed
        old_cursor_pos = pgn.tell();

        # Read ONLY the headers instead of the whole game, so we don't do potentially unnecessary parsing 
        headers = chess.pgn.read_headers(pgn)
        if headers is None:
           print(f"Error: headers section isn't present for the {game_counter}. game.")
           break

        # Filter out the games that we don't need
        if headers.get("Termination") != "Normal" or not headers.get("Event", "").startswith("Rated"):
           continue
        
        # Now, if we got here, it means that the game is appropriate for us.
        # However, our cursor already went past it in order to read its headers.
        # Thats why we go back to the old cursor position and parse the whole game.
        pgn.seek(old_cursor_pos)
        game = chess.pgn.read_game(pgn)
        
        # TODO: Extract whatever from the game.
        
        game_counter = game_counter + 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pgn", help="Path to the PGN file", type=Path, required=True)
    parser.add_argument("-s", "--split", help="Specifies which split is the pgn file used for", required=True, choices=["trainval", "test"])
    parser.add_argument("-g", "--games", help="Specifies how many games will be used for the split", type=int, required=True)
    args = parser.parse_args()

    if not args.pgn.is_file():
        parser.error(f"Hello cruel world, {args.pgn} is not a file!")

    build_cache(pgn_path=args.pgn)
   

# TODO:
# 1) Is 'chess' library an overkill for pgn parsing?
