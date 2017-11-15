#!/usr/bin/env python
#
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

import argparse
import configparser
import logging
import os
import sys
import scipy.io as sio

# change the graphic backend in case of no attached display
try:
    import prosolia.plot as plot
except ModuleNotFoundError: # PyQT
    import matplotlib
    matplotlib.use('pdf')

import prosolia.plot as plot
import prosolia.pipeline as pipeline

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

        except KeyboardInterrupt:
            self._exit('keyboard interruption, exiting')


def str2bool(s, safe=False):
    """Return True if s=='true', False if s=='false'

    If s is already a bool return it, else raise TypeError.
    If `safe` is True, never raise but return False instead.

    """
    if isinstance(s, bool):
        return s

    s = s.lower()

    if safe:
        return True if s == 'true' else False
    else:
        if s == 'true':
            return True
        if s == 'false':
            return False
    raise TypeError("{} must be 'true' or 'false'".format(s))


def parse_args(argv=sys.argv[1:]):
    """Return parsed arguments from command-line"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Extract pitch, probability of voicing and '
        'filterbank energy modulation from a wav file')

    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='display log messages to stdout')

    parser.add_argument(
        '-p', '--plot', metavar='<image_file>', nargs='?',
        default=None, const=True,
        help='plot the pipeline result in a file if <image_file> specified. '
        'If -p is used without argument, render the figure to screen')

    parser.add_argument(
        '-o', '--output', metavar='<mat_file>', default=None,
        help='output file in Matlab format, default is <wav>.mat')

    parser.add_argument(
        '--tstart', metavar='<float>', default=None, type=float,
        help='start time (in s.) from where to read the wav file, '
        'if not specified read from the begining')

    parser.add_argument(
        '--tstop', metavar='<float>', default=None, type=float,
        help='stop time (in s.) from where to stop reading the wav file, '
        'if not specified read up to the end')

    parser.add_argument(
        '-c', '--config', type=str, metavar='<config_file>', required=True,
        help='configuration file to load')

    parser.add_argument(
        'wav', nargs=1, metavar='<wav_file>',
        help='input wav file')

    args = parser.parse_args(argv)
    args.wav = args.wav[0]
    args.output = (os.path.splitext(args.wav)[0] + '.mat'
                   if args.output is None else args.output)
    return args


@CatchExceptions
def main(argv=sys.argv[1:]):
    """Entry point of the program when used from command-line"""
    # parse the input arguments and load configuration
    args = parse_args(argv)
    config = configparser.ConfigParser()
    config._interpolation = configparser.ExtendedInterpolation()
    config.read(args.config)

    # setup the log
    log = logging.getLogger('prosolia')
    log.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    log.addHandler(logging.StreamHandler(sys.stdout))

    # load the input audio file
    audio, sample_frequency = pipeline.load_audio(
        args.wav, args.tstart, args.tstop)

    # compute the spectrogram from gammatone filters
    spectrogram, center_frequencies = pipeline.apply_gammatone(
        audio, sample_frequency,
        nb_channels=config.getint('filterbank', 'nb_channels'),
        low_cf=config.getfloat('filterbank', 'low_frequency'),
        window_time=config.getfloat('filterbank', 'window_time'),
        overlap_time=eval(config.get('filterbank', 'overlap_time')),
        compression=config.get('filterbank', 'compression'),
        accurate=str2bool(config.get('filterbank', 'accurate')))

    # compute delta and delta-delta on spectrogram
    spectrogram = {
        'binned': spectrogram,
        'energy': pipeline.apply_energy(spectrogram)}

    # compute pitch and probability of voicing
    pov, pitch = pipeline.apply_pitch(
        config['pitch']['kaldi_root'],
        args.wav, sample_frequency,
        eval(config.get('pitch', 'frame_length')),
        eval(config.get('pitch', 'frame_shift')),
        config.get('pitch', 'options'))

    # save results
    log.info('saving to %s', args.output)
    sio.savemat(args.output, {
        'wav': args.wav,
        'config': config,
        'sample_frequency': sample_frequency,
        'center_frequencies': center_frequencies,
        'spectrogram': spectrogram,
        'pitch': pitch,
        'pov': pov})

    # plot results
    if args.plot:
        log.info('plotting to %s', 'screen' if args.plot is True else args.plot)
        plot.plot_pipeline(
            sample_frequency,
            config.getfloat('filterbank', 'low_frequency'),
            audio, spectrogram, pov, pitch,
            output_file=args.plot if isinstance(args.plot, str) else None)


if __name__ == '__main__':
    main()
