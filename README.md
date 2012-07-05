This is a C++ implementation (with C and Matlab MEX interfaces) of Support
Distribution Machines, as described by:

Barnabas Poczos, Liang Xiong, Dougal J. Sutherland, and Jeff Schneider.
_Nonparametric Kernel Estimators for Image Classification._
CVPR 2012. http://autonlab.org/autonweb/20680.html

The code was written by Dougal J. Sutherland.

A pure-Matlab version of this code (much slower, but easier to set up if you're
a Matlab user), written by Liang Xiong, is available at
    http://autonlab.org/autonweb/20466.html

Requirements
------------

  * [np-divs](https://github.com/dougalsutherland/np-divs/) for nonparametric
    divergence estimation
  * [LibSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) for support vector
    machines
  * [Boost](http://boost.org) at least 1.35
  * [CMake](http://cmake.org)
  * BLAS and LAPACK implementations
  * A working FORTRAN compiler (to use BLAS/LAPACK)

Installation
------------

    mkdir build; cd build
    cmake ..
    make
    make runtests # optional
    make install

This will install the shared library named e.g. `libsdm.so` (depending on
platform) and header files, as well as an `sdm-run` binary and a MEX file to
`share/sdm/matlab`. By default, these will be placed in `/usr/local`; to
install to a different location, use something like:

    cmake .. -DCMAKE_INSTALL_PREFIX=$HOME
