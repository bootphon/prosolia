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
"""Implementation of the prosolia pipeline"""

import logging
import math
import os
import shlex
import shutil
import subprocess
import tempfile

import numpy as np


def load_audio(filename, tstart=None, tstop=None, dtype=np.float64):
    """Return audio data from a file

    The wav file is assumed to be mono.

    Parameters:
    -----------

    filename (string or openfile handle): input audio file

    tstart (float): begin time of the audio chunk to load (in s.),
      None to begin from start.

    tstop (floet): end time of the audio chunk to load (in s.), None to
      load up to the en dof the file.

    dtype: output scalar type (default is numpy.float64)

    Returns:
    --------

    audio (dtype numpy array): data read from file

    sample_frequency (int): sample frequency of the audio content in Hz

    """
    import soundfile
    sample_frequency = soundfile.info(filename).samplerate
    audio, _ = soundfile.read(
        filename, dtype=dtype,
        start=math.floor(tstart * sample_frequency) if tstart else 0,
        stop=math.floor(tstop * sample_frequency) if tstop else None)

    logging.getLogger('prosolia').debug(
        'loaded %s: %ss @ %sHz',
        filename, len(audio)/sample_frequency, sample_frequency)

    return audio, sample_frequency


def apply_gammatone(data, sample_frequency, nb_channels=20, low_cf=20,
                    window_time=0.5, overlap_time=0.1,
                    compression=None, accurate=True):
    """Return the response of a gammatone filterbank to data

    Calculate a spectrogram-like time frequency magnitude array based
    on gammatone subband filters. The waveform ``data`` (at sample
    rate ``sample_frequency``) is passed through an multi-channel
    gammatone auditory model filterbank, with lowest frequency
    ``min_cf`` and highest frequency ``sample_frequency`` / 2. The
    outputs of each band then have their energy integrated over
    windows of ``window_time`` seconds, advancing by ``overlap_time``
    secs for successive columns. The energy is then optionally
    compressed by log10 or cubic root. These magnitudes are returned
    as a nonnegative real matrix with ``nb_channels`` rows (excepted
    for log compression where values in dB are negative).

    Parameters:
    -----------

    data (float numpy array): 1D input data to be processed

    sample_frequency (int): sample frequency of the data in Hz

    nb_channels (int): number of frequency channels in the filterbank

    low_cf (float): lowest center frequency of the filterbank in Hz

    window_time (float): integration time of the window in seconds

    overlap_time (float): overlap time of two successive windows in seconds

    compression (string): compression method to use on energy, choose
        None to disable compression, 'log' for 20*np.log10(X) or
        'cubic' for X**(1/3), default is None

    accurate (bool): use the full filterbank approach instead of the
        weighted FFT approximation. This is much slower, and uses a
        lot of memory, but is more accurate. Default is True.

    Returns:
    --------

    output (float numpy array): 2D filterbank response to the input
        data, where output.shape[0] (time axis) depends on the window
        time and output.shape[1] == nb_channels

    center_frequencies (float numpy array): center frequencies of each
        channel in Hz.

    """
    import gammatone.gtgram
    import gammatone.fftweight
    from gammatone.filters import erb_space

    # choose real gammatones or FFT approximation
    gtgram = (gammatone.gtgram.gtgram if accurate
              else gammatone.fftweight.fft_gtgram)

    logging.getLogger('prosolia').debug(
        'computing filterbank energy on %s channels, %s compression%s',
        nb_channels, compression, ', accurate' if accurate else '')

    # get the center frequencies in increasing order
    center_frequencies = erb_space(
        low_cf, sample_frequency/2, nb_channels)[::-1]

    # get the filterbank output (with increasing frequencies)
    output = np.flipud(gtgram(
        data,
        sample_frequency,
        window_time,
        overlap_time,
        nb_channels,
        low_cf))

    # compress the output
    compress = {'log': lambda X: 20 * np.log10(X),
                'cubic': lambda X: X ** (1./3)}
    try:
        output = compress[compression](output)
    except KeyError:
        pass

    return output, center_frequencies


def apply_delta(array):
    """Compute delta features on a 2D array

    From https://github.com/bootphon/spectral/blob/master/spectral/_spectral.py

    Parameters:
    -----------

    array (2D numpy array): input time/frequency matrix

    Returns:
    --------

    delta: a numpy array such as delta.shape == array.shape

    """
    logging.getLogger('prosolia').debug('computing delta')
    from scipy.signal import lfilter

    X = array.T if array.ndim > 1 else array
    hlen = 4
    a = np.r_[hlen:-hlen-1:-1] / 60
    g = np.r_[np.array([X[1, :] if X.ndim > 1 else X[1] for _ in range(hlen)]),
              X,
              np.array([X[-1, :] if X.ndim > 1 else X[-1] for _ in range(hlen)])]

    res = (lfilter(a, 1, g, axis=0)[hlen:-hlen, :].T if X.ndim > 1 else
           lfilter(a, 1, g, axis=0)[hlen:-hlen])
    return res


def apply_deltadelta(array):
    """Compute delta-delta on a 2D array

    From https://github.com/bootphon/spectral/blob/master/spectral/_spectral.py

    """
    logging.getLogger('prosolia').debug('computing delta-delta')
    from scipy.signal import lfilter

    X = array.T if array.ndim > 1 else array
    hlen = 4
    a = np.r_[hlen:-hlen-1:-1] / 60

    hlen2 = 1
    f = np.r_[hlen2:-hlen2-1:-1] / 2

    g = np.r_[np.array([X[1, :] if X.ndim > 1 else X[1] for _ in range(hlen+hlen2)]),
              X,
              np.array([X[-1, :] if X.ndim > 1 else X[-1] for _ in range(hlen+hlen2)])]

    return (
        lfilter(f, 1, lfilter(
            a, 1, g, axis=0), axis=0)[hlen+hlen2:-hlen-hlen2, :].T
        if X.ndim > 1 else
        lfilter(f, 1, lfilter(
            a, 1, g, axis=0), axis=0)[hlen+hlen2:-hlen-hlen2])


def apply_dct(array, norm=False, size=8):
    """Return the `size` first coefficients of the `array` DCT

    Apply type 2 discrete cosine transfrom on the first axis of `data`
    (frequencies) over the second axis (time). Wrapper on
    scipy.fftpack.dct.

    Parameters:
    -----------

    array (2D numpy array): input array, first axis is frequency,
        second axis is time

    norm: if True, normalize the dct such that makes the corresponding
        matrix of coefficients orthonormal, default is False

    size (int): keep the `size` first coefficients of the dct output

    Return:
    -------

    dct: numpy array of shape (size, data.shape[1])

    """
    logging.getLogger('prosolia').debug('computing DCT')

    from scipy.fftpack import dct
    return dct(
        array, type=2, axis=0,
        norm='ortho' if norm is True else None
    )[:size, :]


def apply_pitch(kaldi_root, wavfile, sample_frequency,
                frame_length=25, frame_shift=10, options=''):
    """Apply Kaldi pitch extractor on a wav file

    Output is 2-dimensional numpy array consisting of (NCCF, pitch in
    Hz), where NCCF is between -1 and 1, and higher for voiced frames.

    Parameters:
    -----------

    kaldi_root (str): path to the root directory of a compiled Kaldi
        distribution. Looks for and executes
        'kaldi_root'/src/featbin/compute-kaldi-pitch-feats

    wavfile (str): path to the wav file to be analyzed

    sample_frequency (int): sampling rate of the input wav file

    frame-length (float): frame length in milliseconds, default is 25

    frame-shift (float): frame shift in milliseconds, default is 10

    options (str): optional parameters to compute-kaldi-pitch-feats


    Options:
    --------

    The following options must be concatened in the `options` string
    as "--key=value" pairs separated by spaces.

    --delta-pitch : Smallest relative change in pitch that our
      algorithm measures (float, default = 0.005)

    --frames-per-chunk : Only relevant for offline pitch extraction
      (e.g. compute-kaldi-pitch-feats), you can set it to a small
      nonzero value, such as 10, for better feature compatibility with
      online decoding (affects energy normalization in the algorithm)
      (int, default = 0)

    --lowpass-cutoff : cutoff frequency for LowPass filter (Hz)
      (float, default = 1000)

    --lowpass-filter-width : Integer that determines filter width of
      lowpass filter, more gives sharper filter (int, default = 1)

    --max-f0 : max. F0 to search for (Hz) (float, default = 400)

    --max-frames-latency : Maximum number of frames of latency that we
      allow pitch tracking to introduce into the feature processing
      (affects output only if --frames-per-chunk > 0 and
      --simulate-first-pass-online=true (int, default = 0)

    --min-f0 : min. F0 to search for (Hz) (float, default = 50)

    --nccf-ballast : Increasing this factor reduces NCCF for quiet
      frames (float, default = 7000)

    --penalty-factor : cost factor for FO change. (float, default =
      0.1)

    --resample-frequency : Frequency that we down-sample the signal to.
      Must be more than twice lowpass-cutoff (float, default = 4000)

    --snip-edges : If this is set to false, the incomplete frames near
      the ending edge won't be snipped, so that the number of frames
      is the file size divided by the frame-shift. This makes
      different types of features give the same number of
      frames. (bool, default = true)

    --soft-min-f0 : Minimum f0, applied in soft way, must not exceed
      min-f0 (float, default = 10)

    --upsample-filter-width : Integer that determines filter width
      when upsampling NCCF (int, default = 5)

    Raise:
    ------

    AssertionError if compute-kaldi-pitch-feats executable is not
    found in the Kaldi tree

    RuntimeError if compute-kaldi-pitch-feats failed

    """
    logging.getLogger('prosolia').debug('estimating pitch and POV')

    # locate the kaldi executable we want
    kaldi_pitch = os.path.join(
        kaldi_root, 'src', 'featbin', 'compute-kaldi-pitch-feats')
    assert os.path.isfile(kaldi_pitch), '{} not found'.format(kaldi_pitch)

    try:
        # directory where kaldi read and write
        tempdir = tempfile.mkdtemp()

        # register wav input to kaldi
        scp = os.path.join(tempdir, 'wav.scp')
        with open(scp, 'w') as fscp:
            fscp.write('{} {}\n'.format(
                os.path.splitext(os.path.basename(wavfile))[0],
                os.path.abspath(wavfile)))

        # the kaldi pitch/pov output
        pitch = os.path.join(tempdir, 'pitch.txt')

        # the kaldi command to execute
        command = (
            kaldi_pitch + ' --sample-frequency={0} --frame-length={1} '
            '--frame-shift={2} {3} scp:{4} ark,t:{5}'
            .format(sample_frequency, frame_length, frame_shift,
                    options, scp, pitch))

        # execute it in a kaldi environment, ignore kaldi log messages
        job = subprocess.Popen(
            shlex.split(command),
            cwd=tempdir, stderr=open(os.devnull))
        job.wait()
        if job.returncode != 0:
            raise RuntimeError('command "{}" returned with {}'
                               .format(command, job.returncode))

        # return the result as two numpy vectors
        a = np.loadtxt(pitch, skiprows=1, usecols=(0, 1))
        return a[:, 0].T, a[:, 1].T

    finally:
        shutil.rmtree(tempdir)
