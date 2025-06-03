from scipy.signal import butter, filtfilt, resample
import scipy
import numpy as np
import matplotlib.pyplot as plt
import os


def plot(sample, figsize_x=10, figsize_y=10, label=""):
    plt.figure(figsize=(figsize_x, figsize_y), )
    plt.plot(sample, label=label)
    if label: plt.legend()
    plt.show()
    
    
def plot_spectrogram(spectrogram, extent, figsize_x=10, figsize_y=10):
    # TODO min max normalization
    min = np.min(spectrogram)
    max = np.max(spectrogram)
    
    s = 10*np.log10(spectrogram)
    
    fig, axx = plt.subplots(1,1, figsize=(figsize_x, figsize_y), sharex='all', sharey='all')
    t_lo, t_hi, f_lo, f_hi = extent
    
    axx.set_xlim(t_lo, t_hi)
    axx.set_ylim(f_lo, f_hi)
    
    kw = dict(origin='lower', aspect='auto', cmap='viridis')
    im = axx.imshow( s, extent=extent, **kw)
    fig.colorbar(im, ax=axx, label="PSD $|S_z(t, f)|^2$")
    fig.show()


def stft(data, sampling_freq):
    stft_window = 'hann'
    stft_nperseg = 128
    stft_noverlap = int(stft_nperseg - 7)
    stft_obj = scipy.signal.ShortTimeFFT.from_window(stft_window, sampling_freq, stft_nperseg, stft_noverlap, scale_to='magnitude')
    
    Sx = stft_obj.stft(data)
    extent = stft_obj.extent(data.shape[0], center_bins=True)
    # delta_f = stft_obj.delta_f
    delta_t = stft_obj.delta_t
    
    # remove time values outside the original window range
    i = 0
    t_lo, t_hi, _, _ = extent
    while t_lo < 0:
        i += 1
        t_lo += delta_t
        t_hi -= delta_t
    Sx = Sx[:, i:-i]
    extent = (t_lo, t_hi, extent[2], extent[3])
    
    return Sx, extent


def compute_and_plot_spectrogram(sample, sampling_freq):
    Sx, extent = stft(sample, sampling_freq)
    spectrogram = abs(Sx)**2
    plot_spectrogram(spectrogram, extent, figsize_x=50, figsize_y=5)

    
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