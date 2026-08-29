#!/usr/bin/env python3
import sys, argparse
from pathlib import Path
import chess.pgn


def build_cache(pgn_path: Path):
   pass 




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pgn", help="Path to the PGN file")
    parser.add_argument("-s", "--split", help="Specifies which split is the pgn file used for")
    parser.add_argument("-g", "--games", help="Specifies how many games will be used for the split")
    args = parser.parse_args()

    if not args.pgn or not args.split or not args.games:
        print("hello cruel world")
    
    pgn_path: Path = Path(args.pgn)

    with open(pgn_path) as pgn:
        first_game = chess.pgn.read_game(pgn)

        print("First game: ", first_game)

        build_cache(pgn_path)



# TODO:
# 1) Is 'chess' library an overkill for pgn parsing?
# 2) We could have a small helper function for reading .pgn.zst files instead of having to decompress manually
