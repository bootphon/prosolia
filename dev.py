#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np

from prosolia.wav import load_wav
from prosolia.filterbank import apply_gammatone
from prosolia.plot import plot_filterbank, plot_audio


# load the input audio file
audiofile = './test/data/test.wav'
data, sample_frequency = load_wav(audiofile)

plt.subplot(3, 1, 1)
plot_audio(data, sample_frequency)


# compute the gammatone response
nb_channels = 20
low_cf = 20
window_time = 0.05
overlap_time = window_time / 2
output, center_frequencies = apply_gammatone(
    data, sample_frequency, nb_channels=nb_channels, low_cf=low_cf,
    window_time=window_time, overlap_time=overlap_time)

# plot cubic compression
output_root = output ** (1./3)
plt.subplot(3, 1, 2)
time=np.linspace(0, len(data) / sample_frequency, num=len(output[0]))
plt.gca().set_xlim([0, time[-1]])
plt.gca().set_ylim([output_root.min(), output_root.max()])
for n, channel in enumerate(output):
    plt.plot(time, output_root[n])
plt.ylabel('energy (cubic)')

# plot log compression
output_log = 20 * np.log10(output)
plt.subplot(3, 1, 3)
time=np.linspace(0, len(data) / sample_frequency, num=len(output[0]))
plt.gca().set_xlim([0, time[-1]])
plt.gca().set_ylim([output_log.min(), output_log.max()])
for n, channel in enumerate(output):
    plt.plot(time, output_log[n])
plt.ylabel('energy (log)')


# ploot "spectrogram"
plt.subplot(3, 1, 3)
plot_filterbank(sample_frequency, center_frequencies, output_log)

plt.show()

# plot the gammatone output
# fig = plt.figure()
# axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

# gammatone.plot.gtgram_plot(
#     gammatone.gtgram.gtgram,
#     axes,
#     data,
#     samplerate,
#     twin, thop, channels, fmin)

# axes.set_title(wavfile)
# axes.set_xlabel("Time (s)")
# axes.set_ylabel("Frequency")
# plt.show()
