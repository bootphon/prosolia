#!/usr/bin/env python
#
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

import argparse
import gammatone
import matplotlib.pyplot as plt
import numpy
import scipy.io.wavfile
import sys


def parse_args(argv=sys.argv[1:]):
    """Return parsed arguments from command-line"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Extract pitch, probability of voicing and '
        'frequency-band energy modulation from a wav file')

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='display log messages to stdout')

    parser.add_argument(
        'wav',
        nargs=1,
        help='input wav file')

    gt_parser = parser.add_argument_group('gammatone filterbank parameters')
    gt_parser.add_argument(
        '-n', '--nb-channels',
        type=int, metavar='<int>',
        default=20,
        help='number of frequency channels in the filterbank '
        '(default is %(default)s)')

    gt_parser.add_argument(
        '-f', '--frequencies',
        type=float, metavar='<float>',
        nargs=2,
        default=[100, 8000],
        help='lowest/highest center frequencies of the filterbank in Hz '
        '(default is %(default)s)')

    return parser.parse_args(argv)


def load_wav(filename):
    """Return the sample frequency and data from a wav file

    Simple wrapper on scipy.io.wavfile.read

    Parameters:
    -----------

    filename (string or openfile handle): input wav file

    Returns:
    --------

    sample_frequency (int): sample frequency of the wav file in Hz

    data (float numpy array): data read from wav file

    """
    fs, data = scipy.io.wavfile.read(filename)
    return fs, data.astype(numpy.float64)


def apply_gammatone(
        data, sample_frequency,
        nb_channels=20, low_cf=100, high_cf=8000):
    """Return the response of a gammatone filterbank to data

    Wrapper on the gammatone.Filterbank class


    Parameters:
    -----------

    data (float numpy array): 1D input data to be processed

    sample_frequency (int): sample frequency of the data in Hz

    nb_channels (int): number of frequency channels in the filterbank

    low_cf (float): lowest center frequency of the filterbank in Hz

    high_cf (float): highest center frequency of the filterbank in Hz

    Returns:
    --------

    center_frequencies (float numpy array): center frequencies of each
        channel in Hz.

    output (float numpy array): 2D filterbank response to the input data,
        shaped as (data.shape[0], nb_channels).

    """
    fb = gammatone.Filterbank(sample_frequency, low_cf, high_cf, nb_channels)
    return numpy.array(fb.center_frequency), fb.compute(data)


class CatchExceptions(object):
    """A decorator wrapping 'function' in a try/except block

    When an exception occurs, display a user friendly message before
    exiting.

    """
    def __init__(self, function):
        self.function = function

    def _exit(self, msg):
        sys.stderr.write(msg + '\n')
        sys.exit(1)

    def __call__(self):
        try:
            self.function()

        except (IOError, OSError, RuntimeError, AssertionError) as err:
            self._exit('fatal error: {}'.format(err))

        # except subprocess.CalledProcessError as err:
        #     self._exit('subprocess fatal error: {}'.format(err))

        except KeyboardInterrupt:
            self._exit('keyboard interruption, exiting')


def plot_gammatone(sample_frequency, center_frequencies, data):
    data = data[:100][:].T ** (1.0/3)
    time = numpy.linspace(
        0, data.shape[0] / float(sample_frequency), data.shape[0])

    y = numpy.array([center_frequencies, ] * data.shape[0]).T
    x = numpy.array([time, ] * data.shape[1])

    # print 'y', y.shape
    # print y

    # print 'x', x.shape
    # print x

    # print 'data', data.shape
    # print data

    data_min, data_max = -numpy.abs(data).max(), numpy.abs(data).max()

    plt.imshow(data, cmap='RdBu', vmin=data_min, vmax=data_max,
     #          extent=[x.min(), x.max(), y.min(), y.max()],
               interpolation='nearest', origin='lower')
    #plt.title('image (interp. nearest)')
    plt.colorbar()

    plt.show()

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
    # plt.show()

@CatchExceptions
def main(argv=sys.argv[1:]):
    args = parse_args(argv)

    sample_frequency, data = load_wav(args.wav[0])

    low_cf, high_cf = tuple(sorted(args.frequencies))
    center_frequencies, output = apply_gammatone(
        data, sample_frequency,
        nb_channels=args.nb_channels,
        low_cf=low_cf, high_cf=high_cf)

    plot_gammatone(sample_frequency, center_frequencies, output)

if __name__ == '__main__':
    main()
