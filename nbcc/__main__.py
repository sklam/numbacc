import sys
from pprint import pprint

from .frontend import frontend


def main(argv: list[str]) -> None:
    [filename] = argv
    frontend(filename)


if __name__ == "__main__":
    main(sys.argv[1:])
