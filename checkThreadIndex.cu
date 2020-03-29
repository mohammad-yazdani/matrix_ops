#include <stdio.h>
#include <cuda_runtime.h>

/*#define CHECK(call)
{
    const cudaError_t error = call;
    if (error != cudaSuccess)
    {
        printf("Error: %s:%d, ", __FILE__, __LINE__);
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));
        exit(1);
    }
}*/

void initialInt(int *ip, int size) {
    for (int i = 0; i < size; i++) {
        ip[i] = i;
    }
}

void printMatrix(int *C, const int nx, const int ny) {
    int *ic = C;
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            printf("%3d", ic[j]);
        }
        ic += nx;
        printf("\n");
    }
    printf("\n");
}

__global__ void printThreadIndex(int *A, const int nx, const int ny) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = ix + (iy * nx);

    printf("thread_id (%d, %d) block_id (%d, %d) coordinates (%d, %d) global index %2d ival %2d\n", threadIdx.x, threadIdx.y, 
        blockIdx.x, blockIdx.y, ix, iy, idx, A[idx]);
}

void devConfig(const int devId) {
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, devId);
    printf("Using Device %d: %s\n", devId, devProp.name);
    cudaSetDevice(devId);
}

int main(int argc, char *argv[]) {
    devConfig(0);

    int nx = 8;
    int ny = 6;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    int *h_A;
    h_A = (int *)malloc(nBytes);

    initialInt(h_A, nxy);
    printMatrix(h_A, nx, ny);

    int *d_MatA;
    cudaMalloc((void **)&d_MatA, nBytes);

    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);

    dim3 block(4, 2);
    dim3 grid((nx + (block.x - 1)) / block.x, (ny + (block.y - 1)) / block.y);

    printThreadIndex <<< grid, block >>>(d_MatA, nx, ny);
    cudaDeviceSynchronize();

    cudaFree(d_MatA);
    free(h_A);

    cudaDeviceReset();

    return (0);
}
