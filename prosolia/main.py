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
import matplotlib.pyplot as plt
import sys

import filterbank
import plot
import wav


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


@CatchExceptions
def main(argv=sys.argv[1:]):
    args = parse_args(argv)

    data, sample_frequency = wav.load_wav(args.wav[0])

    plt.subplot(211)
    plot.plot_audio(data, sample_frequency)

    low_cf, high_cf = tuple(sorted(args.frequencies))
    center_frequencies, output = filterbank.apply_gammatone(
        data, sample_frequency,
        nb_channels=args.nb_channels,
        low_cf=low_cf, high_cf=high_cf)

    plt.subplot(212)
    plot.plot_filterbank(
        sample_frequency, center_frequencies, output)

    plt.show()

if __name__ == '__main__':
    main()
