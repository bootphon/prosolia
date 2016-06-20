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


def plot_audio(data, sample_frequency, show=False):
    time=np.linspace(0, len(data)/sample_frequency, num=len(data))
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    plt.gca().set_xlim([0, time[-1]])
    plt.plot(time, data)

    if show:
        plt.show()


def plot_filterbank(sample_frequency, center_frequencies, data, show=False):
    data = data[7000:8000].T #** 0.5
    time = np.linspace(
        0, data.shape[0] / float(sample_frequency), data.shape[0])

    y = np.array([center_frequencies, ] * data.shape[0]).T
    x = np.array([time, ] * data.shape[1])

    plt.imshow(data,
               #cmap='RdBu',
               vmin=data.min(), vmax=data.max(),
               aspect='auto',
               #extent=[x.min(), x.max(), y.min(), y.max()],
               interpolation='bicubic', origin='lower')
    #plt.title('image (interp. nearest)')
    plt.colorbar()

    # #x = data
    # data = data[:100][:]
    # print data.shape
    # print data
    # interp = 'bilinear'
    # #interp = 'nearest'
    # lim = 0, data.shape[0], 0, data.shape[1]
    # #plt.subplot(211, axisbg='g')
    # #plt.title('blue should be up')
    # plt.imshow(data, origin='upper', interpolation=interp, cmap='jet')
    # plt.axis(lim)
    # plt.subplot(212, axisbg='y')
    # plt.title('blue should be down')
    # plt.imshow(x, origin='lower', interpolation=interp, cmap='jet')
    # #plt.axis(lim)
