# LDLt_factorization_GPU
The course project for "Programmation en GPU" @ ENSAE.

This project aims to implement a batch parallel version of [LDLt factorization](https://en.wikipedia.org/wiki/Cholesky_decomposition) on GPU.

In order to execute the code in `main.cu`, you should have a GPU in your device with __CUDA__ installed.

You can compile the code by the command line `nvcc main.cu -arch=sm_50 -o main` and then execute `./main`.