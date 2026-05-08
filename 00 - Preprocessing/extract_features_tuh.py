"""
Unified TUH feature extraction pipeline.

This script provides a unified entry point for TUH EEG feature extraction,
supporting both background and photostimulation data sources.

Example:
    python extract_features_tuh.py --source both --feature-types spectral cwt
"""

import argparse
import os
import sys

# Ensure the workspace root is on sys.path so local packages (like `utils`) are importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import feature_extraction as fe

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified TUH feature extraction for background and photostimulation data."
    )
    parser.add_argument(
        "--source",
        choices=["background", "ips", "both"],
        default="both",
        help="Which sample set to process.",
    )
    parser.add_argument(
        "--input-root",
        default=fe.DEFAULT_INPUT_ROOT,
        help="Directory containing the input pickle files.",
    )
    parser.add_argument(
        "--output-root",
        default=fe.DEFAULT_OUTPUT_ROOT,
        help="Directory where feature arrays will be written.",
    )
    parser.add_argument(
        "--montages",
        nargs="+",
        default=fe.DEFAULT_MONTAGES,
        choices=fe.DEFAULT_MONTAGES,
        help="Montages to evaluate.",
    )
    parser.add_argument(
        "--segment-lengths",
        nargs="+",
        type=int,
        default=fe.DEFAULT_SEGMENT_LENGTHS,
        help="Segment lengths, in seconds.",
    )
    parser.add_argument(
        "--feature-types",
        nargs="+",
        default=fe.DEFAULT_FEATURE_TYPES,
        choices=fe.DEFAULT_FEATURE_TYPES,
        help="Feature families to extract.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip combinations that already exist on disk.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sources = [args.source] if args.source != "both" else ["background", "ips"]

    for source in sources:
        print("=" * 70)
        print(f"Processing source: {source}")
        print("=" * 70)
        fe.process_source(
            source=source,
            input_root=args.input_root,
            output_root=args.output_root,
            montages=args.montages,
            segment_lengths=args.segment_lengths,
            feature_types=args.feature_types,
            skip_existing=args.skip_existing,
        )


if __name__ == "__main__":
    main()
