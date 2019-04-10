// This is a GPU implementation of LDLt factorization

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
// #include "timer.h"

__global__ void LDLt_max_k(float *A, float *y, int d) {
    int tidx = threadIdx.x % d; // Thread identifier in a grid for solving one linear system
    int Qt = (threadIdx.x - tidx) / d; // Local grid identifier for one linear system
    int gbx = Qt + blockIdx.x * (blockDim.x / d); // Global grid identifier 

    extern __shared__ float sA[]; // shared memory to accelarate the computation

    int i, k, grid, nt, n2;

    grid = d*(d+1)/2 + d;
    n2 = grid;
    nt = Qt * grid;

    // The d-th thread in a grid collects matrix and vector from global to shared memory
    if (tidx == d-1) {
        int offset = -d;
        for (i = 0; i < d; i++) {
            offset += d - i;
            for (k = i; k < d; k++) {
                sA[nt + d + offset + k] = A[gbx*d*d + k*d + i];
            }
        }
        for (i = 0; i < d; i++){
            sA[nt + i] = y[gbx*d + i];
        }
    }

    // // Verifier that the data is really copied from global memory to shared memory
    // if (gbx == 199 && tidx == d-1) {
    //     int offset = -d;
    //     for (i = 0; i < d; i++) {
    //         offset += d - i;
    //         for (k = 0; k < i; k++) {
    //             printf("%f\t", 0.0f);
    //         }
    //         for (k = i; k < d; k++) {
    //             printf("%f\t", sA[nt + d + offset + k]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    //     for (i = 0; i < d; i++){
    //         printf("%f\t", sA[nt + i]);
    //     }
    //     printf("\n");
    //     printf("\n");
    // }
    __syncthreads(); // Wait for the collection finishing

    // Perform the LDLt factorization
    for (i = d; i > 0; i--) {
        // The first thread in a grid computes D_i
        if (tidx == 0) {
            for (k = d; k > i; k--) {
                sA[nt + n2 - i*(i+1)/2] -= sA[nt + n2 - k*(k+1)/2] * sA[nt + n2 - k*(k+1)/2 + k - i] * sA[nt + n2 - k*(k+1)/2 + k - i];
            }
        }
        __syncthreads(); // Wait for finishing D_i computation
        // The first i-1 threads in a grid compute L_ji
        if (tidx < i-1) {
            sA[nt + n2 - i*(i+1)/2 + tidx + 1] /= sA[nt + n2 - i*(i+1)/2];
            for (k = d; k > i; k--) {
                sA[nt + n2 - i*(i+1)/2 + tidx + 1] -= sA[nt + n2 - k*(k+1)/2] * sA[nt + n2 - k*(k+1)/2 + k - i] * sA[nt + n2 - k*(k+1)/2 + tidx + 1 + k - i] / sA[nt + n2 - i*(i+1)/2];
            }
        }
        __syncthreads(); // Wait for finishing L_ji computation
    }

    // Solving the linear system using LDLt factorization
    if (tidx == d-1) {
        for (i = d; i > 0; i--) {
            for (k = d; k > i; k--) {
                sA[nt + d - i] -= sA[nt + d - k] * sA[nt + n2 - k*(k+1)/2 + k - i];
            }
            sA[nt + d - i] /= sA[nt + n2 - i*(i+1)/2];
        }
        for (i = 0; i < d; i++) {
            for (k = 0; k < i; k++) {
                sA[nt + d - i - 1] -= sA[nt + d - k - 1] * sA[nt + n2 - i*(i+1)/2 - k - 1]; // The final solution
            }
            y[gbx*d + d - i - 1] = sA[nt + d - i - 1]; // Copy the final solution from shared memory to global memory
        }
    }
    
}

// NB: The shared memory of Tesla P100-PCIE-16GB is 48KB
// The data to be copied to shared memory should not exceed this size
int main() {
    int i, j, k;
    int dim = 64; // The size of the matrix
    int minTB = 5; // The number of grids per block
    int NB = 16384; // The number of blocks
    int size = NB * minTB; // The number of matrices to be factorized
    float rho; // Parameters to fill the matrices
    float *A, *AGPU, *Y, *YGPU; // The matrices and the vectors both on CPU and GPU

    // Timer Tim; // CPU timer instructions
    // GPU timer instructions
    float TimVar;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Memory allocation
    A = (float *)malloc(size*dim*dim*sizeof(float));
    Y = (float *)malloc(size*dim*sizeof(float));
    cudaMalloc(&AGPU, size*dim*dim*sizeof(float));
    cudaMalloc(&YGPU, size*dim*sizeof(float));

    // Setting matrix and vector values
    for (i = 0; i < size; i++) {
        rho = 1.0f / (1.1f + i);
        for (j = 0; j < dim; j++) {
            for (k = 0; k < dim; k++) {
                if (j == k) {
                    A[i*dim*dim + j*dim + k] = 1.0f;
                }
                else {
                    A[i*dim*dim + j*dim + k] = rho;
                }
            }
            Y[j + i*dim] = 0.5f * j;
        }
    }

    // i = 199;
    // for (j = 0; j < dim; j++) {
    //     for (k = 0; k < dim; k++) {
    //         printf("%f\t", A[i*dim*dim + j*dim + k]);
    //     }
    //     printf("\n");
    // }
    // for (j = 0; j < dim; j++) {
    //     printf("%f\t", Y[i*dim + j]);
    // }
    // printf("\n");
    // printf("\n");

    // Copy matrices and vectors from cpu to GPU
    cudaMemcpy(AGPU, A, size*dim*dim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(YGPU, Y, size*dim*sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(start, 0); // GPU timer starts

    // Solve the linear systems using LDLt factorization on GPU
    LDLt_max_k<<<NB, dim*minTB, minTB*(dim*(dim+1)/2 + dim)*sizeof(float)>>>(AGPU, YGPU, dim);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&TimVar, start, stop); // Get GPU time
    cudaEventDestroy(start); // destroy the GPU timer
    cudaEventDestroy(stop);

    // Copy result from GPU to CPU
    cudaMemcpy(Y, YGPU, size*dim*sizeof(float), cudaMemcpyDeviceToHost);

    // Check a numerical result
    i = 199;
    for (j = 0; j < dim; j++) {
        printf("%f\t", Y[i*dim + j]);
    }
    printf("\n");

    printf("GPU Timer: %f ms\n", TimVar);

    // Memory free
    free(A);
    cudaFree(AGPU);
    free(Y);
    cudaFree(YGPU);

    return 0;
}