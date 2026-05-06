"""
Unified TUH feature extraction pipeline.

This script replaces the separate background and photostimulation extractors
with a single entry point that can process either source or both.
"""

import argparse
import os
import pickle as pkl
import sys

import numpy as np

from utils import feature_extraction_funcs as yf


DEFAULT_INPUT_ROOT = "/space/gzanardini/tuh_eeg/preprocessed"
DEFAULT_OUTPUT_ROOT = "/space/gzanardini/tuh_background"
DEFAULT_BACKGROUND_FILE = "no_photostim_periods.pkl"
DEFAULT_IPS_FILE = "stim_samples.pkl"
DEFAULT_FS = 250
DEFAULT_MONTAGES = ["CAR", "Cz", "BipolarDB", "Laplacian"]
DEFAULT_SEGMENT_LENGTHS = [1, 2, 5, 10, 20, 60, 120]
DEFAULT_FEATURE_TYPES = [
    "spectral",
    "cwt",
    "dwt",
    "mst",
    "sst",
    "utm",
    "plv",
    "gplv",
    "cc",
    "gcc",
]

SOURCE_FILES = {
    "background": DEFAULT_BACKGROUND_FILE,
    "ips": DEFAULT_IPS_FILE,
}


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def sample_to_array(sample):
    if hasattr(sample, "get_data"):
        return sample.get_data()
    return sample


def load_samples(source, input_root):
    file_name = SOURCE_FILES[source]
    file_path = os.path.join(input_root, file_name)

    with open(file_path, "rb") as handle:
        samples = pkl.load(handle)

    if not samples:
        raise ValueError(f"No samples found in {file_path}")

    return samples, file_path


def resolve_sampling_frequency(source, samples):
    if source == "ips":
        return int(samples[0].info["sfreq"])
    return DEFAULT_FS


def run_feature(feature_type, data, fs, montage, segment_length):
    if feature_type == "spectral":
        return yf.run_spectral_seg2(data, fs, MONTAGE=montage, sec=segment_length)
    if feature_type == "cwt":
        return yf.run_cwt_seg(
            data,
            Fs=fs,
            MONTAGE=montage,
            WAVELET_TYPE="morl",
            sec=segment_length,
        )
    if feature_type == "dwt":
        return yf.run_dwt_seg(
            data,
            Fs=fs,
            MONTAGE=montage,
            WAVELET="db4",
            sec=segment_length,
        )
    if feature_type == "mst":
        return yf.run_mST_seg(data, Fs=fs, MONTAGE=montage, epoch_width=segment_length)
    if feature_type == "sst":
        return yf.run_sST_seg(data, Fs=fs, MONTAGE=montage, epoch_width=segment_length)
    if feature_type == "utm":
        return yf.run_UTM_seg(data, Fs=fs, MONTAGE=montage, sec=segment_length)
    if feature_type == "plv":
        return yf.run_plv_seg(data, Fs=fs, MONTAGE=montage, sec=segment_length)
    if feature_type == "gplv":
        with HiddenPrints():
            return yf.run_gplv_seg(data, Fs=fs, MONTAGE=montage, sec=segment_length)
    if feature_type == "cc":
        return yf.run_cc_seg(data, Fs=fs, MONTAGE=montage, sec=segment_length)
    if feature_type == "gcc":
        with HiddenPrints():
            return yf.run_gcc_seg(data, Fs=fs, MONTAGE=montage, sec=segment_length)

    raise ValueError(f"Unsupported feature type: {feature_type}")


def process_source(source, input_root, output_root, montages, segment_lengths, feature_types, skip_existing=False):
    samples, input_path = load_samples(source, input_root)
    fs = resolve_sampling_frequency(source, samples)
    feature_path = os.path.join(output_root, source)
    os.makedirs(feature_path, exist_ok=True)

    print(f"Loaded {len(samples)} samples from {input_path}")
    print(f"Sampling frequency: {fs} Hz")
    print(f"Saving features to: {feature_path}")

    for feature_type in feature_types:
        for montage in montages:
            for segment_length in segment_lengths:
                output_file = os.path.join(
                    feature_path,
                    f"{feature_type}_{montage}_{segment_length}s.npy",
                )

                if skip_existing and os.path.exists(output_file):
                    print(f"Skipping existing file: {output_file}")
                    continue

                print(f"Starting {feature_type} - {source} - {montage} {segment_length}s")
                extracted_features = []

                for index, sample in enumerate(samples):
                    print(f"Processing sample {index + 1}/{len(samples)}")
                    data = sample_to_array(sample)
                    extracted_features.append(
                        run_feature(feature_type, data, fs, montage, segment_length)
                    )

                extracted_features = np.array(extracted_features)
                print(f"Finished {feature_type} - {source} - {montage} {segment_length}s")
                print(f"Shape: {extracted_features.shape}")
                np.save(output_file, extracted_features)


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
        default=DEFAULT_INPUT_ROOT,
        help="Directory containing the input pickle files.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where feature arrays will be written.",
    )
    parser.add_argument(
        "--montages",
        nargs="+",
        default=DEFAULT_MONTAGES,
        choices=DEFAULT_MONTAGES,
        help="Montages to evaluate.",
    )
    parser.add_argument(
        "--segment-lengths",
        nargs="+",
        type=int,
        default=DEFAULT_SEGMENT_LENGTHS,
        help="Segment lengths, in seconds.",
    )
    parser.add_argument(
        "--feature-types",
        nargs="+",
        default=DEFAULT_FEATURE_TYPES,
        choices=DEFAULT_FEATURE_TYPES,
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
        process_source(
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