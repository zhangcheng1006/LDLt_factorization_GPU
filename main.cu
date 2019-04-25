// This is a GPU implementation of LDLt factorization

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "timer.h"

// CPU implementation of LDLt factorization for i-th linear system
// In order for result verification
void LDLt(float *A, float *y, int d, int i) {
    float L[d][d];
    float D[d];
    float b[d], z[d];
    int j, k, l;

    // Perform LDLt factorization
    for (j = 0; j < d; j++) {
        D[j] = A[i*d*d + j*d + j];
        for (k = 0; k < j; k++) {
            D[j] -= L[j][k] * L[j][k] * D[k];
        }
        for (k = j+1; k < d; k++) {
            L[k][j] = A[i*d*d + k*d + j] / D[j];
            for (l = 0; l < j; l++) {
                L[k][j] -= L[k][l] * L[j][l] * D[l] / D[j];
            }
        }
    }

    // Solve the linear system
    for (j = 0; j < d; j++) {
        b[j] = y[j];
        for (k = 0; k < j; k++) {
            b[j] -= L[j][k] * b[k];
        }
        z[j] = b[j] / D[j];
    }
    for (j = d-1; j >= 0; j--) {
        y[j] = z[j];
        for (k = d-1; k > j; k--) {
            y[j] -=  y[k] * L[k][j];
        }
    }
}

// Function to check if the linear system is corrected solved on GPU
// The error is defined as '''mean((y1 - y2)^2)'''
float Check_Result(float *y1, float *y2, int d) {
    int i;
    float error = 0.0f;
    for (i = 0; i < d; i++) {
        error += (y1[i] - y2[i]) * (y1[i] - y2[i]);
    }
    error /= d;
    return error;
}

// LDLt_max_k kernel: perform batch LDLt factorization on GPU by row calculation
// For d <= 64, there is enough shared memory for all the data of a linear system
// So we copy all the data to the shared memory to accelerate the calculation
__global__ void LDLt_max_k(float *A, float *y, int d) {
    int tidx = threadIdx.x % d; // Thread identifier in a grid for solving one linear system
    int Qt = (threadIdx.x - tidx) / d; // Local grid identifier for one linear system
    int gbx = Qt + blockIdx.x * (blockDim.x / d); // Global grid identifier 

    extern __shared__ float sA[]; // shared memory to accelarate the computation

    int i, k, grid, nt, n2;

    grid = d*(d+1)/2 + d;
    n2 = grid;
    nt = Qt * grid;

    // The d threads in a grid parallelly collect matrix and vector from global to shared memory
    // // version 1: for each iteration, take the exact number of threads to copy one row of the matrix
    // //            need d iterations
    // for (i = d; i > 0; i--) {
    //     if (tidx < i) {
    //         sA[nt + n2 - i*(i+1)/2 + tidx] = A[gbx*d*d + (d-i)*d + tidx + d-i];
    //     }
    // }
    // sA[nt + tidx] = y[gbx*d + tidx];
    // __syncthreads(); // Wait for the collection finishing

    // version 2: for each iteration, all threads can work together to read two rows of the matrix
    // //         need d/2 iterations
    // //         and if there are even numbers of rows, need to handle the d/2-th row
    for (i = d; i > d/2; i--) {
        if (tidx < i) {
            sA[nt + n2 - i*(i+1)/2 + tidx] = A[gbx*d*d + (d-i)*d + tidx + d-i];
        }
        else {
            sA[nt + n2 - (d-i)*(d-i+1)/2 + tidx - i] = A[gbx*d*d + i*d + tidx];
        }
    }
    if (d % 2 == 0) {
        if (tidx < d/2) {
            sA[nt + n2 - d/2*(d/2+1)/2 + tidx] = A[gbx*d*d + (d/2)*d + tidx + d/2];
        }
    }
    sA[nt + tidx] = y[gbx*d + tidx];
    __syncthreads(); // Wait for the collection finishing
    // ### There is no great different between these two versions of copying data

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
    for (i = d-1; i > 0; i--) {
        if (tidx < i) {
            sA[nt + d - i + tidx] -= sA[nt + d - i - 1] * sA[nt + n2 - (i+1)*(i+2)/2 + tidx + 1];
        }
        __syncthreads(); // Waiting for the previous finishing
    }
    sA[nt + tidx] /= sA[nt + n2 - (d-tidx)*(d-tidx+1)/2];
    __syncthreads(); // Waiting for all threads finishing
    for (i = 1; i < d; i++) {
        if (tidx < d-i) {
            sA[nt + tidx] -= sA[nt + d - i] * sA[nt + n2 - (d-tidx)*(d-tidx+1)/2 + d-tidx-i];
        }
        __syncthreads(); // Waiting for the previous finishing
    }
    y[gbx*d + tidx] = sA[nt + tidx];
}

// LDLt_max_col_k kernel: perform batch LDLt factorization on GPU by column calculation
// In theory, this will be a little slower than the row version
// Because the closer threads do not read closer cases in the global memory
// In practice, as d <= 64 is small and we access to global memory only once for copying data
// and the data are stored the same in shared memory, so there is no great difference
__global__ void LDLt_max_col_k(float *A, float *y, int d) {
    int tidx = threadIdx.x % d; // Thread identifier in a grid for solving one linear system
    int Qt = (threadIdx.x - tidx) / d; // Local grid identifier for one linear system
    int gbx = Qt + blockIdx.x * (blockDim.x / d); // Global grid identifier 

    extern __shared__ float sA[]; // shared memory to accelarate the computation

    int i, k, grid, nt, n2;

    grid = d*(d+1)/2 + d;
    n2 = grid;
    nt = Qt * grid;

    // The d thread in a grid parallelly collect matrix and vector from global to shared memory
    // // version 1
    // for (i = d; i > 0; i--) {
    //     if (tidx < i) {
    //         sA[nt + n2 - i*(i+1)/2 + tidx] = A[gbx*d*d + (tidx + d-i)*d + d-i];
    //     }
    // }
    // sA[nt + tidx] = y[gbx*d + tidx];
    // __syncthreads(); // Wait for the collection finishing

    // version 2
    for (i = d; i > d/2; i--) {
        if (tidx < i) {
            sA[nt + n2 - i*(i+1)/2 + tidx] = A[gbx*d*d + (tidx + d-i)*d + d-i];
        }
        else {
            sA[nt + n2 - (d-i)*(d-i+1)/2 + tidx - i] = A[gbx*d*d + tidx*d + i];
        }
    }
    if (d % 2 == 0) {
        if (tidx < d/2) {
            sA[nt + n2 - d/2*(d/2+1)/2 + tidx] = A[gbx*d*d + (tidx + d/2)*d + d/2];
        }
    }
    sA[nt + tidx] = y[gbx*d + tidx];
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
    for (i = d-1; i > 0; i--) {
        if (tidx < i) {
            sA[nt + d - i + tidx] -= sA[nt + d - i - 1] * sA[nt + n2 - (i+1)*(i+2)/2 + tidx + 1];
        }
        __syncthreads(); // Waiting for the previous finishing
    }
    sA[nt + tidx] /= sA[nt + n2 - (d-tidx)*(d-tidx+1)/2];
    __syncthreads(); // Waiting for all threads finishing
    for (i = 1; i < d; i++) {
        if (tidx < d-i) {
            sA[nt + tidx] -= sA[nt + d - i] * sA[nt + n2 - (d-tidx)*(d-tidx+1)/2 + d-tidx-i];
        }
        __syncthreads(); // Waiting for the previous finishing
    }
    y[gbx*d + tidx] = sA[nt + tidx];
    
}

// LDLt_max_col_k_2: column calculation but with different data organization in shared memory
// The fastest
__global__ void LDLt_max_col_k_2(float *A, float *y, int d) {
    int tidx = threadIdx.x % d; // Thread identifier in a grid for solving one linear system
    int Qt = (threadIdx.x - tidx) / d; // Local grid identifier for one linear system
    int gbx = Qt + blockIdx.x * (blockDim.x / d); // Global grid identifier 

    extern __shared__ float sA[]; // shared memory to accelarate the computation

    int i, k, grid, nt, n2;

    grid = d*(d+1)/2 + d;
    n2 = d;
    nt = Qt * grid;

    // The d thread in a grid parallelly collect matrix and vector from global to shared memory
    // // version 1
    // for (i = 0; i < d; i++) {
    //     if (tidx <= i) {
    //         sA[nt + n2 + i*(i+1)/2 + tidx] = A[gbx*d*d + i*d + tidx];
    //     }
    // }
    // sA[nt + tidx] = y[gbx*d + tidx];
    // __syncthreads(); // Wait for the collection finishing

    // version 2
    for (i = d; i > d/2; i--) {
        if (tidx >= d-i) {
            sA[nt + n2 + (i-1)*i/2 + tidx-d+i] = A[gbx*d*d + (i-1)*d + tidx-d+i];
        }
        else {
            sA[nt + n2 + (d-i-1)*(d-i)/2 + tidx] = A[gbx*d*d + (d-i-1)*d + tidx];
        }
    }
    if (d % 2 == 0) {
        if (tidx < d/2) {
            sA[nt + n2 + (d/2-1)*d/2/2 + tidx] = A[gbx*d*d + (d/2-1)*d + tidx];
        }
    }
    sA[nt + tidx] = y[gbx*d + tidx];
    __syncthreads(); // Wait for the collection finishing

    // Perform the LDLt factorization
    for (i = 0; i < d; i++) {
        // The first thread in a grid computes D_i
        if (tidx == 0) {
            for (k = 0; k < i; k++) {
                sA[nt + n2 + i*(i+1)/2 + i] -= sA[nt + n2 + k*(k+1)/2 + k] * sA[nt + n2 + i*(i+1)/2 + k] * sA[nt + n2 + i*(i+1)/2 + k];
            }
        }
        __syncthreads(); // Wait for finishing D_i computation

        // The first i-1 threads in a grid compute L_ji
        if (tidx > i) {
            sA[nt + n2 + tidx*(tidx+1)/2 + i] /= sA[nt + n2 + i*(i+1)/2 + i];
            for (k = 0; k < i; k++) {
                sA[nt + n2 + tidx*(tidx+1)/2 + i] -= sA[nt + n2 + k*(k+1)/2 + k] * sA[nt + n2 + i*(i+1)/2 + k] * sA[nt + n2 + tidx*(tidx+1)/2 + k] / sA[nt + n2 + i*(i+1)/2 + i];
            }
        }
        __syncthreads(); // Wait for finishing L_ji computation
    }

    // Solving the linear system using LDLt factorization
    for (i = 0; i < d; i++) {
        if (tidx > i) {
            sA[nt + tidx] -= sA[nt + i] * sA[nt + n2 + tidx*(tidx+1)/2 + i];
        }
        __syncthreads(); // Waiting for the previous finishing
    }
    sA[nt + tidx] /= sA[nt + n2 + tidx*(tidx+1)/2 + tidx];
    __syncthreads(); // Waiting for all threads finishing
    for (i = d-1; i > 0; i--) {
        if (tidx < i) {
            sA[nt + tidx] -= sA[nt + i] * sA[nt + n2 + i*(i+1)/2 + tidx];
        }
        __syncthreads(); // Waiting for the previous finishing
    }
    y[gbx*d + tidx] = sA[nt + tidx];
    
}

// LDLt_large_k: Batch LDLt factorization for large d > 64
// We can no more copy all the data to the shared memory because of memory limitation
// But we can still copy the vector y and diagonal elements to shared memory
__global__ void LDLt_large_k(float *A, float *y, int d) {
    int tidx = threadIdx.x % d; // Thread identifier in a grid for solving one linear system
    int Qt = (threadIdx.x - tidx) / d; // Local grid identifier for one linear system
    int gbx = Qt + blockIdx.x * (blockDim.x / d); // Global grid identifier 

    extern __shared__ float sA[];

    int i, k, grid, nt, n2, gbn;

    float res;

    grid = 2 * d;
    nt = Qt * grid;
    n2 = d;
    gbn = gbx * d * d;

    // copy y and diagonal elements to shared memory
    sA[nt + tidx] = y[gbx*d + tidx];
    sA[nt + n2 + tidx] = A[gbn + tidx*d + tidx];
    __syncthreads();

    // Perform the LDLt factorization
    for (i = 0; i < d; i++) {
        // The first thread in a grid computes D_i
        if (tidx == 0) {
            for (k = 0; k < i; k++) {
                sA[nt + n2 + i] -= A[gbn + k*d + i] * A[gbn + k*d + i] * sA[nt + n2 + k];
            }
        }
        __syncthreads(); // Wait for finishing D_i computation

        // The first i-1 threads in a grid compute L_ji
        if (tidx > i) {
            res = A[gbn + i*d + tidx]; // use register to avoid accessing to the global memory several times
            res /= sA[nt + n2 + i];
            for (k = 0; k < i; k++) {
                res -= sA[nt + n2 + k] * A[gbn + k*d + tidx] * A[gbn + k*d + i] / sA[nt + n2 + i];
            }
            A[gbn + i*d + tidx] = res;
        }
        __syncthreads(); // Wait for finishing L_ji computation
    }

    // Solving the linear system using LDLt factorization
    for (i = 1; i < d; i++) {
        if (tidx >= i) {
            sA[nt + tidx] -= sA[nt + i - 1] * A[gbn + (i-1)*d + tidx];
        }
        __syncthreads(); // Waiting for the previous finishing
    }
    sA[nt + tidx] /= sA[nt + n2 + tidx];
    __syncthreads(); // Waiting for all threads finishing
    for (i = d-1; i > 0; i--) {
        if (tidx < i) {
            sA[nt + tidx] -= sA[nt + i] * A[gbn + tidx*d + i];
        }
        __syncthreads(); // Waiting for the previous finishing
    }
    y[gbx*d + tidx] = sA[nt + tidx];
}

// LDLt_large_col_k: As before, perform the column version for large d
// In this case, the column version is much slower than the row version
// as we take d = 1024 is large and we should access global memory many times for calculation
__global__ void LDLt_large_col_k(float *A, float *y, int d) {
    int tidx = threadIdx.x % d; // Thread identifier in a grid for solving one linear system
    int Qt = (threadIdx.x - tidx) / d; // Local grid identifier for one linear system
    int gbx = Qt + blockIdx.x * (blockDim.x / d); // Global grid identifier 

    extern __shared__ float sA[];

    int i, k, grid, nt, n2, gbn;

    float res;

    grid = 2 * d;
    nt = Qt * grid;
    n2 = d;
    gbn = gbx * d * d;

    // copy y and diagonal elements to shared memory
    sA[nt + tidx] = y[gbx*d + tidx];
    sA[nt + n2 + tidx] = A[gbn + tidx*d + tidx];
    __syncthreads();

    // Perform the LDLt factorization
    for (i = 0; i < d; i++) {
        // The first thread in a grid computes D_i
        if (tidx == 0) {
            for (k = 0; k < i; k++) {
                sA[nt + n2 + i] -= A[gbn + i*d + k] * A[gbn + i*d + k] * sA[nt + n2 + k];
            }
        }
        __syncthreads(); // Wait for finishing D_i computation

        // The first i-1 threads in a grid compute L_ji
        if (tidx > i) {
            res = A[gbn + tidx*d + i];
            res /= sA[nt + n2 + i];
            for (k = 0; k < i; k++) {
                res -= sA[nt + n2 + k] * A[gbn + tidx*d + k] * A[gbn + i*d + k] / sA[nt + n2 + i];
            }
            A[gbn + tidx*d + i] = res;
        }
        __syncthreads(); // Wait for finishing L_ji computation
    }

    // Solving the linear system using LDLt factorization
    for (i = 1; i < d; i++) {
        if (tidx >= i) {
            sA[nt + tidx] -= sA[nt + i - 1] * A[gbn + tidx*d + i-1];
        }
        __syncthreads(); // Waiting for the previous finishing
    }
    sA[nt + tidx] /= sA[nt + n2 + tidx];
    __syncthreads(); // Waiting for all threads finishing
    for (i = d-1; i > 0; i--) {
        if (tidx < i) {
            sA[nt + tidx] -= sA[nt + i] * A[gbn + i*d + tidx];
        }
        __syncthreads(); // Waiting for the previous finishing
    }
    y[gbx*d + tidx] = sA[nt + tidx];
}

// NB: The shared memory of Tesla P100-PCIE-16GB is 48KB
// The data to be copied to shared memory should not exceed this size
int main() {
    int i, j, k;

    // setting for testing LDLt_max_k, LDLt_max_col_k and LDLt_max_col_k_2
    int dim = 64; // The size of the matrix
    int minTB = 5; // The number of grids per block
    int NB = 16384; // The number of blocks

    // // setting for testing LDLt_large_k, LDLt_large_col_k
    // int dim = 1024; 
    // int minTB = 1;
    // int NB = 219;


    int size = NB * minTB; // The number of matrices to be factorized
    float rho; // Parameters to fill the matrices
    float *A, *AGPU, *Y, *YGPU; // The matrices and the vectors both on CPU and GPU

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

    printf("Memory Allocated!\n");
    printf("%i\n", size*dim*dim);

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

    printf("Matrices and Vectors set!\n");

    // Check one numerical result in CPU
    i = 29;
    float x[dim];
    for (j = 0; j < dim; j++) {
        x[j] = Y[i*dim + j];
    }

    Timer Tim; // CPU timer instructions

    Tim.start(); // CPU timer instructions

    LDLt(A, x, dim, i);

    Tim.add(); // CPU timer instructions

    // print numerical results
    for (j = 0; j < dim; j++) {
        printf("%f\t", x[j]);
    }
    printf("\n\n");

    printf("CPU Timer for one linear system: %f ms\n\n", (float)Tim.getsum()*1000); // CPU timer instructions
    // Estimate the time for solving all linear systems
    printf("Estimation of CPU Time for solving all linear systems: %f s\n\n", (float)Tim.getsum()*size);

    // Copy matrices and vectors from cpu to GPU
    cudaMemcpy(AGPU, A, size*dim*dim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(YGPU, Y, size*dim*sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(start, 0); // GPU timer starts

    // Solve the linear systems using LDLt factorization on GPU
    // LDLt_max_k<<<NB, dim*minTB, minTB*(dim*(dim+1)/2 + dim)*sizeof(float)>>>(AGPU, YGPU, dim);
    // LDLt_max_col_k<<<NB, dim*minTB, minTB*(dim*(dim+1)/2 + dim)*sizeof(float)>>>(AGPU, YGPU, dim);
    LDLt_max_col_k_2<<<NB, dim*minTB, minTB*(dim*(dim+1)/2 + dim)*sizeof(float)>>>(AGPU, YGPU, dim);
    // LDLt_large_k<<<NB, dim*minTB, minTB*dim*2*sizeof(float)>>>(AGPU, YGPU, dim);
    // LDLt_large_col_k<<<NB, dim*minTB, minTB*dim*2*sizeof(float)>>>(AGPU, YGPU, dim);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&TimVar, start, stop); // Get GPU time
    cudaEventDestroy(start); // destroy the GPU timer
    cudaEventDestroy(stop);

    // Copy result from GPU to CPU
    cudaMemcpy(Y, YGPU, size*dim*sizeof(float), cudaMemcpyDeviceToHost);

    // Check the selected numerical result
    for (j = 0; j < dim; j++) {
        printf("%f\t", Y[i*dim + j]);
    }
    printf("\n\n");

    printf("GPU Timer: %f ms\n\n", TimVar);

    float error;
    error = Check_Result(x, Y+i*dim, dim);
    printf("The calculation error between CPU and GPU is: %f\n", error);

    // Memory free
    free(A);
    cudaFree(AGPU);
    free(Y);
    cudaFree(YGPU);

    return 0;
}