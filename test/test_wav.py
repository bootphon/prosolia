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
"""Test of prosolia.wav"""

import numpy as np
import os
import prosolia.wav

HERE = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(HERE, 'data', 'test.wav')


def test_wav():
    data, fs = prosolia.wav.load_wav(filename)

    assert fs == 20000
    assert np.max(data) <= 1.0
    assert np.min(data) >= -1.0
