#include <stdio.h>

// Function that catches the error 
void testCUDA(cudaError_t error, const char *file, int line)  {

	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d\n", file, line);
       exit(EXIT_FAILURE);
	} 
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

__global__ void empty_k(void){
	// printf("Hello World!\n");
	printf("thread idx %d, block idx %d\n", threadIdx.x, blockIdx.x);
}

int main (void){

	// threads are synchronized by group of 32
	empty_k<<<8,2>>>(); // <number of block, number of thread per block>
	cudaDeviceSynchronize();

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	printf("Number of GPUs: %d\n", deviceCount);

	cudaDeviceProp deviceProp;
	// testCUDA(cudaGetDeviceProperties(&deviceProp, deviceCount)); // will return error
	cudaGetDeviceProperties(&deviceProp, deviceCount-1);
	printf("Device %d has compute capability %d.%d.\n",
				deviceCount-1, deviceProp.major, deviceProp.minor);
	printf("Name: %s\n", deviceProp.name);
	printf("Number of processors: %d\n", 128*deviceProp.multiProcessorCount);
	printf("GPU RAM size in bytes: %zd\n", deviceProp.totalGlobalMem);
	printf("Shared memory per block in bytes: %zd\n", deviceProp.sharedMemPerBlock);

	/*************************************************************

	Once requested, replace this comment by the appropriate code

	*************************************************************/


	return 0;
}