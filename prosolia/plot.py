# Copyright 2016 Mathieu Bernard
#
# You can redistribute this file and/or modify it under the terms of
# the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
"""Plotting functions for the prosalia pipeline"""

import math
import matplotlib.pyplot as plt
import numpy as np


def plot_pipeline(sample_frequency, low_frequency, audio,
                  spectrogram, dct_output, pov, pitch):
    """Plot the whole prosalia output as 4 subplots

    Displays audio signal (plot 1), probability of voicing and pitch
    estimation (plot 2), filterbank output (plot 3) and DCT output
    (plot 4).

    """
    fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=6, sharex=True)
    fig.subplots_adjust(wspace=0, hspace=0)

    plot_audio(ax0, audio, sample_frequency)

    plot_pitch(ax1, len(audio)/sample_frequency, pov, pitch)

    plot_filterbank(
        fig, ax2, sample_frequency, low_frequency,
        len(audio)/sample_frequency, spectrogram['raw'], label='spectrogram')

    plot_filterbank(
        fig, ax3, sample_frequency, low_frequency,
        len(audio)/sample_frequency, spectrogram['delta'], label='delta')

    plot_filterbank(
        fig, ax4, sample_frequency, low_frequency,
        len(audio)/sample_frequency, spectrogram['delta_delta'], label='delta delta')

    plot_dct(fig, ax5, len(audio)/sample_frequency, dct_output)

    ax5.set_xlabel('time (s)')
    fig.tight_layout()
    plt.show()


def plot_audio(axes, data, sample_frequency):
    """Plot the audio signal"""
    time = np.linspace(0, len(data)/sample_frequency, num=len(data))
    axes.set_ylabel('amplitude')
    axes.set_xlim([0, time[-1]])
    axes.plot(time, data)


def plot_pitch(axes, duration, pov, pitch):
    """Plot the pitch estimation and probability of voicing"""
    time = np.linspace(0, duration, num=len(pov))

    par1 = axes.twinx()
    p1, = axes.plot(time, pov, 'b', label="NCCF")
    p2, = par1.plot(time, pitch['raw'], 'r', ls='-', label="pitch (Hz)")
    # p3, = par1.plot(time, pitch['delta'] * 10, 'r', ls='--', label="pitch delta")
    # p4, = par1.plot(time, pitch['delta_delta'] * 100, 'r', ls='-.', label="pitch delta-delta")

    axes.set_xlim(0, duration)
    axes.set_ylim(-1, 1)

    # round max pitch to the upper hundredth for nice plotting
    par1.set_ylim(0, int(math.ceil(pitch['raw'].max() / 100.0)) * 100)

    axes.set_ylabel("NCCF")
    par1.set_ylabel("pitch (Hz)")
    axes.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())

    tkw = dict(size=4, width=1.5)
    axes.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)


def plot_filterbank(fig, axes, sample_frequency, low_cf, duration,
                    data, label=''):
    """Plot the filterbank output as an image with colorbar"""
    # Set a nice formatter for the y-axis
    from gammatone.plot import ERBFormatter
    formatter = ERBFormatter(low_cf, sample_frequency/2, unit='Hz', places=0)
    axes.yaxis.set_major_formatter(formatter)

    img = axes.imshow(data, extent=[0, duration, 1, 0], aspect='auto')
    axes.set_ylabel('frequency' if not label else label)
    # fig.colorbar(img, ax=axes)


def plot_dct(fig, axes, duration, data):
    """Plot the DCT output as an image with colorbar"""
    # n as in pipeline.apply_dct
    n = data.shape[0]

    axes.set_ylabel("DCT")
    axes.set_yticks(range(n, 0, -1))
    img = axes.imshow(data, extent=[0, duration, 0, n], aspect='auto',
                      origin='lower', interpolation='nearest')
    # fig.colorbar(img, ax=axes)
