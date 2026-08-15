"""Compatibility entry point for the maintained dependency-free Phi bounds."""

from perbacco.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["reproduce", "--scope", "table-iii"]))
