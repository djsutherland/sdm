This is a C++ implementation of Support Distribution Machines, as described by:

Barnabas Poczos, Liang Xiong, Dougal J. Sutherland, and Jeff Schneider, 2012.
_Support Distribution Machines._
Technical report, Carnegie Mellon University.
[arXiv:1202.0302](http://arxiv.org/abs/1202.0302).

The code was written by Dougal J. Sutherland.

Requirements
------------

  * [np-divs](https://github.com/dougalsutherland/np-divs/) for nonparametric
    divergence estimation
  * [LibSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) for support vector
    machines
  * [Boost](http://boost.org)
  * [CMake](http://cmake.org)
  * BLAS and LAPACK implementations

Installation
------------

    mkdir build; cd build
    cmake ..
    make
    make runtests # optional, requires HDF5
    make install

This will install the shared library named e.g. `libsdm.so` (depending on
platform) and header files. By default, these will be placed in `/usr/local`;
to install to a different location, use something like:

    cmake .. -DCMAKE_INSTALL_PREFIX=$HOME
