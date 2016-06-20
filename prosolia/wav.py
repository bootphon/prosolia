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
"""Load wav files as numpy arrays"""

import numpy as np
import soundfile as sf

def load_wav(filename, dtype=np.float64):
    """Return audio data from a wav file

    The wav file is assumed to be mono.

    Parameters:
    -----------

    filename (string or openfile handle): input wav file

    dtype: output scalar type (default is numpy.float64)

    Returns:
    --------

    sample_frequency (int): sample frequency of the wav file in Hz

    data (dtype numpy array): data read from wav file

    """
    return sf.read(filename, dtype=dtype)
