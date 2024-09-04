from scipy.signal import spectrogram, butter, filtfilt, resample
import numpy as np
import matplotlib.pyplot as plt
import os

# TODO update
def plot_image(img):
    plt.figure(figsize=(10, 6))
    # plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', vmin=vmin, vmax=vmax)
    plt.imshow(img, cmap='plasma')
    # plt.colorbar(label='Power Spectral Density (dB)')
    # plt.ylabel('Frequency (Hz)')
    # plt.xlabel('Time (s)')
    # plt.title('Spectrogram')
    plt.show()
    
# TODO update
def plot_spectrogram(f, t, Sxx, vmin=-200, vmax=0):
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Power Spectral Density (dB)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Spectrogram')
    plt.show()

    # TODO update
def compute_and_plot_spectrogram(data, sf, vmin=-200, vmax=0):
    f, t, Sxx = spectrogram(data, fs=sf, nperseg=256)
    plot_spectrogram(f, t, Sxx, vmin, vmax)

# TODO update
def spectrogram_to_image(Sxx, vmin=-200, vmax=0):
    Sxx_log = 10 * np.log10(Sxx)
    Sxx_norm = (Sxx_log - vmin) / (vmax - vmin)
    Sxx_norm = np.clip(Sxx_norm, 0, 1)
    return (Sxx_norm * 255).astype(np.uint8) # to gray scale (0-255)

    
def notch_filter(data, sf, cutoff_freq=60, order=4, quality_factor=1):
    nyquist = 0.5 * sf
    low = (cutoff_freq - quality_factor) / nyquist
    high = (cutoff_freq + quality_factor) / nyquist
    b, a = butter(order, [low, high], btype='bandstop') # filter coefficients
    return filtfilt(b, a, data) # apply filter # TODO check if it is the best option    


def highpass_filter(data, sf, cutoff_freq=0.5, order=4):
    nyquist = 0.5 * sf
    high = cutoff_freq / nyquist
    b, a = butter(order, high, btype='highpass') # filter coefficients
    return filtfilt(b, a, data) # apply filter # TODO check if it is the best option


def downsample(data, old_sf, new_sf):
    len_in_seconds = int(data.shape[-1] / old_sf)
    nr_samples_after_downsampling = len_in_seconds * new_sf
    downsampled_data = resample(data, nr_samples_after_downsampling, axis=-1)
    return downsampled_data


def folder_exists(folder_path):
    return os.path.exists(folder_path) and os.path.isdir(folder_path)

def file_exists(file_path):
    return os.path.exists(file_path) and os.path.isfile(file_path)