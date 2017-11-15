# Copyright 2016, 2017 Mathieu Bernard
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
                  spectrogram, pov, pitch,
                  output_file=None):
    """Plot the whole prosalia output as 4 subplots

    If `output_file` is specified, write the plot the that file (image
    format guessed from extension), else render it to screen.

    Displays audio signal (plot 1), probability of voicing and pitch
    estimation (plot 2), filterbank output (plot 3) and filterbank
    energy (plot 4).

    """
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, sharex=True)
    fig.subplots_adjust(wspace=0, hspace=0)

    plot_audio(ax0, audio, sample_frequency)

    plot_pitch(ax1, len(audio)/sample_frequency, pov, pitch)

    plot_filterbank(
        fig, ax2, sample_frequency, low_frequency,
        len(audio) / sample_frequency, spectrogram['binned'],
        label='spectrogram')

    plot_energy(ax3, len(audio)/sample_frequency, spectrogram['energy'])

    ax3.set_xlabel('time (s)')
    fig.tight_layout()

    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()


def plot_audio(axes, data, sample_frequency, label='amplitude'):
    """Plot the audio signal"""
    time = np.linspace(0, len(data)/sample_frequency, num=len(data))
    axes.set_ylabel(label)
    axes.set_xlim([0, time[-1]])
    axes.plot(time, data)


def plot_energy(axes, duration, energy):
    """Plot the filterbank energy over time"""
    time = np.linspace(0, duration, num=len(energy))
    axes.set_ylabel('energy')
    axes.set_xlim([0, time[-1]])
    axes.plot(time, energy)

def plot_pitch(axes, duration, pov, pitch):
    """Plot the pitch estimation and probability of voicing"""
    time = np.linspace(0, duration, num=len(pov))

    par1 = axes.twinx()
    p1, = axes.plot(time, pov, 'b', label="NCCF")
    p2, = par1.plot(time, pitch, 'r', ls='-', label="pitch (Hz)")

    axes.set_xlim(0, duration)
    axes.set_ylim(-1, 1)

    # round max pitch to the upper hundredth for nice plotting
    par1.set_ylim(0, int(math.ceil(pitch.max() / 100.0)) * 100)

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
