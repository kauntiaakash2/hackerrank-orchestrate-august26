"""Command-line interface for deterministic message routing."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .src.data_loader import DatasetBundle
from .src.router import Router
from .src.validation import validate_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("dataset/output.csv"))
    parser.add_argument("--cache", type=Path, default=Path("cache/media_extractions.json"))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--audit", type=Path, help="Write a machine-generated data audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    data = DatasetBundle.load(args.dataset_dir)
    if args.audit:
        args.audit.write_text(data.audit(), encoding="utf-8")
    if not args.validate_only:
        Router(data, args.cache).route_all().to_csv(args.output, index=False)
        logging.info("wrote %s", args.output)
    validate_output(args.output, data)
    logging.info("output validation passed")


if __name__ == "__main__":
    main()
