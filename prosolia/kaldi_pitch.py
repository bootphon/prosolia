# Copyright 2016 Thomas Schatz, Xuan Nga Cao, Mathieu Bernard
#
# This file is part of abkhazia: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Abkhazia is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with abkhazia. If not, see <http://www.gnu.org/licenses/>.

import numpy
import os
import shlex
import shutil
import subprocess
import sys
import tempfile


def _run(command, cwd=None, verbose=False):
    """Run 'command' as a subprocess

    command : string to be executed as a subprocess

    cwd : current working directory for executing the command

    verbose : if True, forward messages on stderr

    Returns silently if the command returned with 0, else raise a
    RuntimeError

    """
    stderr = None if verbose else open(os.devnull)
    job = subprocess.Popen(shlex.split(command), cwd=cwd, stderr=stderr)
    job.wait()

    if job.returncode != 0:
        raise RuntimeError('command "{}" returned with {}'
                           .format(command, job.returncode))


def apply_pitch(kaldi_root, wavfile, sample_frequency, verbose=True):
    """Apply Kaldi pitch extractor on a wav file

    Output is 2-dimensional features consisting of (NCCF, pitch in
    Hz), where NCCF is between -1 and 1, and higher for voiced frames.

    Raise:
    ------

    AssertionError if compute-kaldi-pitch-feats executable is not
    found in the Kaldi tree

    RuntimeError if compute-kaldi-pitch-feats failed

    """
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
        command = (kaldi_pitch + ' --sample-frequency={0} scp:{1} ark,t:{2}'
                   .format(sample_frequency, scp, pitch))

        # execute it in a kaldi environment
        _run(command, cwd=tempdir, verbose=verbose)

        # return the result as two numpy vectors
        a = numpy.loadtxt(pitch, skiprows=1, usecols=(0, 1))
        return a[:,0].T, a[:,1].T

    finally:
        shutil.rmtree(tempdir)
