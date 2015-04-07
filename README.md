# LAS-CUDA
CUDA implementation of [Large Average Submatrix algorithm](https://projecteuclid.org/euclid.aoas/1254773275#info) for GPU-accelerated search of, well, large-on-average submatrices.

You will need:
* NVIDIA<sup>®</sup> Nsight Eclipse Edition
* GCC of some version with C++11 support
* [googletest](https://code.google.com/p/googletest/) if you want to do some testing.

Open project in Nsight, let it generate makefiles, build *Release* configuration 
and you'll find binaries and headers ready for you to use in **Release/dist** 
directory. 

If you want some testing, get yourself into **Test/data** directory first,
run every single *&#42;Generator.m* file with [octave](https://www.gnu.org/software/octave/),
build *Test* configuration and execute **Test/LAS**. I suggest insalling
[this](https://github.com/xgsa/cdt-tests-runner/wiki/Tutorial) plugin
to see test results directly in Nsight.
