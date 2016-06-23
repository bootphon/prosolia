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
import os
import scipy.io as sio
import sys

import pipeline
import plot


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
        '-v', '--verbose', action='store_true',
        help='display log messages to stdout')

    parser.add_argument(
        '--plot', action='store_true',
        help='display the pipeline result as plots')

    parser.add_argument(
        'wav', nargs=1, help='input wav file')

    parser.add_argument(
        '--output', nargs=1, default=None, help='optional .mat output file')

    gt_parser = parser.add_argument_group('gammatone filterbank parameters')
    gt_parser.add_argument(
        '-n', '--nb-channels',
        type=int, metavar='<int>',
        default=30,
        help='number of frequency channels in the filterbank '
        '(default is %(default)s)')

    gt_parser.add_argument(
        '-l', '--low-frequency',
        type=float, metavar='<float>', default=20,
        help='lowest center frequency of the filterbank in Hz '
        '(default is %(default)s Hz)')

    en_parser = parser.add_argument_group('energy parameters')
    en_parser.add_argument(
        '-w', '--window-time', metavar='<float>', type=float, default=0.08,
        help='integration time of the energy window in seconds, '
        'default is %(default)s s')

    en_parser.add_argument(
        '-o', '--overlap-time', metavar='<float>', type=float, default=None,
        help='overlap time of two successive windows in seconds, '
        'default is "window-time / 2"')

    en_parser.add_argument(
        '-c', '--compression', choices=['no', 'cubic', 'log'], default='no',
        help='type of energy compression, "no" disable compression (default), '
        '"cubic" is cubic root and "log" is 20*log10')

    pi_parser = parser.add_argument_group('pitch and POV parameters')
    pi_parser.add_argument(
        '-k', '--kaldi-root', default='../kaldi',
        help='root directory of the Kaldi distribution, default is %(default)s')

    args = parser.parse_args(argv)

    args.wav = args.wav[0]

    if args.output is None:
        args.output = os.path.splitext(args.wav)[0] + '.mat'

    if args.overlap_time is None:
        args.overlap_time = args.window_time / 2

    return args


def save( filename, args, sample_frequency,
          center_frequencies, energy, dct):
    sio.savemat(filename, {
        'wav': args.wav,
        'sample_frequency': sample_frequency,
        'center_frequencies': center_frequencies,
        'window_time': args.window_time,
        'overlap_time': args.overlap_time,
        'compression': args.compression,
        'energy': energy,
        'dct': dct})


@CatchExceptions
def main(argv=sys.argv[1:]):
    """Entry point of the program when used from command-line"""
    # parse the input arguments
    args = parse_args(argv)

    # load the input audio file
    if args.verbose:
        print('reading from {}'.format(args.wav))
    audio, sample_frequency = pipeline.load_audio(args.wav)

    # compute filterbank energy
    energy, center_frequencies = pipeline.apply_gammatone(
        audio, sample_frequency,
        nb_channels=args.nb_channels, low_cf=args.low_frequency,
        window_time=args.window_time, overlap_time=args.overlap_time,
        compression=args.compression)

    # compute DCT on energy
    dct_output = pipeline.apply_dct(energy)

    # compute pitch and probability of voicing
    from kaldi_pitch import apply_pitch
    pov, pitch = apply_pitch(args.kaldi_root, args.wav, sample_frequency, args.verbose)

    # save results
    if args.verbose:
        print('saving to {}'.format(args.output))
    save(args.output, args, sample_frequency, center_frequencies,
         energy, dct_output)

    if args.plot:
        if args.verbose:
            print('plotting...')
        plot.plot_pipeline(sample_frequency, args.low_frequency, audio,
                           energy, dct_output, pov, pitch)

if __name__ == '__main__':
    main()
