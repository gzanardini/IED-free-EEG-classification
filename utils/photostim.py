from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def round_to_odd(value: float) -> int:
    return int(round((value + 1) / 2) * 2 - 1)


def extract_photostim_chunks(
    photostim_periods,
    wins_df,
    chunk_length: float = 19.3,
    fs: int = 250,
    start_sample: int = 100,
    skip_indices: tuple[int, ...] = (35,),
    anchor_channel: int = 19,
    trim_samples: int = 2550,
    channel_offset: int = 25,
):
    chunk_samples = int(chunk_length * fs)
    photostim_chunks = []
    chunk_rows = []

    for i, period in enumerate(photostim_periods):
        if i in skip_indices:
            continue

        for t in range(start_sample, period.shape[1], chunk_samples):
            if t + chunk_samples <= period.shape[1]:
                temp = period[:, t:t + chunk_samples]
            else:
                temp = period[:, t:]

            if np.all(temp[anchor_channel] == 0):
                continue

            non_zero = np.flatnonzero(np.abs(temp[anchor_channel]) > 0)
            if non_zero.size == 0:
                continue

            start = max(int(non_zero[0]) - channel_offset, 0)
            end = start + trim_samples
            temp = temp[:, start:end].copy()

            numpeaks = find_peaks(np.abs(temp[anchor_channel]))[0].shape[0]
            stim_freq = numpeaks / (temp.shape[1] / fs)
            stim_freq_round = round_to_odd(stim_freq)

            photostim_chunks.append(temp)

            row = wins_df.loc[i].copy()
            row['frequency'] = stim_freq_round
            chunk_rows.append(row)

    chunks_df = pd.DataFrame(chunk_rows).reset_index(drop=True)
    if 'frequency' not in chunks_df.columns:
        chunks_df['frequency'] = pd.Series(dtype=float)

    return photostim_chunks, chunks_df