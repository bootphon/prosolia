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
"""Plotting function for the prosalia pipeline"""

import matplotlib.pyplot as plt
import numpy as np

from gammatone.plot import ERBFormatter
from mpl_toolkits.axes_grid1 import ImageGrid


def plot_pipeline(sample_frequency, low_frequency, audio, energy, dct_output, pov, pitch):
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4)
    plot_audio(ax0, audio, sample_frequency)

    plot_pitch(fig, ax1, len(audio)/sample_frequency, pov, pitch)

    plot_filterbank(
        fig, ax2, sample_frequency, low_frequency,
        len(audio)/sample_frequency, energy)

    plot_dct(
        fig, ax3, len(audio)/sample_frequency, dct_output)

    fig.tight_layout()
    plt.show()



def plot_audio(axes, data, sample_frequency):
    time=np.linspace(0, len(data)/sample_frequency, num=len(data))
    axes.set_xlabel('time (s)')
    axes.set_ylabel('amplitude')
    axes.set_xlim([0, time[-1]])
    axes.plot(time, data)


def _roundup(x):
    import math
    return int(math.ceil(x / 100.0)) * 100


def plot_pitch(fig, axes, duration, pov, pitch):
    time = np.linspace(0, duration, num=len(pov))

    par1 = axes.twinx()
    p1, = axes.plot(time, pov, 'b', label="NCCF")
    p2, = par1.plot(time, pitch, 'r', label="pitch (Hz)")

    axes.set_xlim(0, duration)
    axes.set_ylim(-1, 1)
    par1.set_ylim(0, _roundup(pitch.max()))

    axes.set_xlabel("time (s)")
    axes.set_ylabel("NFCC")
    par1.set_ylabel("pitch (Hz)")
    axes.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())

    tkw = dict(size=4, width=1.5)
    axes.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)


def plot_filterbank(fig, axes, sample_frequency, low_cf, duration, data):
    grid = ImageGrid(
        fig, (4, 1, 3),
        nrows_ncols=(1, 1),
        direction="row",
        axes_pad=0.05,
        add_all=True,
        label_mode="1",
        share_all=True,
        cbar_location="right",
        cbar_mode="each",
        cbar_size="7%",
        cbar_pad="1%",
    )

    # Set a nice formatter for the y-axis
    formatter = ERBFormatter(low_cf, sample_frequency/2, unit='Hz', places=0)
    grid[0].yaxis.set_major_formatter(formatter)

    img = grid[0].imshow(data, extent=[0, duration, 1, 0], aspect='auto')
    grid[0].set_xlabel("time (s)")
    grid[0].set_ylabel("frequency")

    fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #fig.colorbar(img, cax=cbar_ax)

    #cbar = gri.cax.colorbar(img, ax=axes)
    # cbar.set_label('energy')


def plot_dct(fig, axes, duration, data):
    # n as in pipeline.apply_dct
    n = data.shape[0]

    axes.set_xlabel("time (s)")
    axes.set_ylabel("DCT coefs")
    axes.set_yticks(range(n, 0, -1))
    img = axes.imshow(data, extent=[0, duration, 0, n], aspect='auto',
                      origin='lower', interpolation='nearest')
    fig.colorbar(img, ax=axes)
