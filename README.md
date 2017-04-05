# Prosolia

**speech features extraction pipeline for prosody analysis**

``` text
                                       +--->  delta        +
                                       |                   |
wav +---> filterbank +---> compression +--->  delta-delta  |
    |                                  |                   +---> .mat file
    |                                  +--->  DCT          |
    |                                                      |
    +---> pitch (with deltas), probability of voicing      +
```

* wav file as input, Matlab mat file as output

* gammatone filterbank (spectrogram-like from the
  [Gammatone Filterbank Toolkit](https://github.com/detly/gammatone))

* compressed spectrogram (cubic root or log)

* delta, delta-delta and dicrete cosine transform computed on spectrogram

* pitch estimation and probability of voicing (from
  [Kaldi](http://kaldi-asr.org))


## Usage

Once installed, prosolia is available as a command-line tool. Pipeline
parameters are read from a configuration file. To get in, simply have
a `prosolia --help`.

For exemple:

``` shell
prosolia ./some_speech.wav -c ./prosolia.cfg -o some_features.mat
```

## Installation

* On Windows, we recommend to install and use prosolia in a
  [CygWin](https://cygwin.com) environment, because Kaldi is better
  supported there.

* Install Kaldi by following the instructions from
  [here](http://kaldi-asr.org/doc/install.html). Note that you don't
  need the whole Kaldi system but only the
  [compute-kaldi-pitch-feats](
  https://github.com/kaldi-asr/kaldi/blob/master/src/featbin/compute-kaldi-pitch-feats.cc)
  program.

  To compile only the required subpart of Kaldi (the `featbin`
  sources), you have to do (from the `kaldi` directory):

``` shell
cd tools
./extras/check_dependencies.sh
make -j 4  # -j N do a parallel build on N CPUs
cd ../src
./configure
make depend -j 4
make featbin -j 4  # use "make -j 4" to compile the entire Kaldi
```

* Prosalia relies on the system library `libsndfile`. On Windows and
  OS X, it is installed automatically. On Linux, you need to install
  libsndfile using your distribution's package manager, for example
  `sudo apt-get install libsndfile1`.

* Using the [Anaconda](http://continuum.io/downloads) distribution of
  Python 3 (conda is also available independently of Anaconda with
  `pip install conda; conda init`):

``` shell
conda install python=3.6 cffi numpy scipy matplotlib
python setup.py install
```

## Licence

**Copyright 2016 Mathieu Bernard**

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
