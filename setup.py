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
"""Installation script for prosolia"""

import setuptools

VERSION = '0.1'

setuptools.setup(
    name='prosolia',
    version=VERSION,
    author='Mathieu Bernard',
    author_email='mmathieubernardd@gmail.com',
    license='GPL3',

    description='speech features extraction pipeline for prosody analysis',
    long_description=open('README.md').read(),

    dependency_links=['https://github.com/detly/gammatone/archive/master.zip#egg=gammatone-1.0'],
    install_requires=[
        'gammatone',
        'matplotlib',
        'numpy',
        'scipy',
        'cffi',
        'pysoundfile'],

    packages=setuptools.find_packages(),
    entry_points={'console_scripts': ['prosolia = prosolia.main:main']},

)
