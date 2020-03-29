#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>

void
sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int i = 0; i < ny; i++)
    {
        for (int j = 0; j < nx; j++)
        {
            ic[j] = ia[j] + ib[j];
        }
        ia += nx;
        ib += nx;
        ic += nx;
    }
}

__global__ void
sumMatrixOnGPU(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
    unsigned int ix = threadIdx.x + (blockIdx.x * blockDim.x);
    unsigned int iy = threadIdx.y + (blockIdx.y * blockDim.y);
    unsigned int idx = ix + (iy * nx);

    if (ix < nx && iy < ny) MatC[idx] = MatA[idx] + MatB[idx];
    else printf("WHAT: ix >= nx || iy >= ny\n");
}

void devConfig(const int devId) {
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, devId);
    printf("Using Device %d: %s\n", devId, devProp.name);
    cudaSetDevice(devId);
}

void
initalData(float *matrix, const int nxy)
{
    for (int i = 0; i < nxy; i++)
        matrix[i] = 0;
}

int
main(int argc, char *argv[])
{
    devConfig(0);

    int nx = 1 << 14;
    int ny = 1 << 14;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A =       (float *)malloc(nBytes);
    h_B =       (float *)malloc(nBytes);
    hostRef =   (float *)malloc(nBytes);
    gpuRef =    (float *)malloc(nBytes);

    clock_t iStart = clock();
    initalData(h_A, nxy);
    initalData(h_B, nxy);
    clock_t iEnd = clock();
    // TODO : Store time

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);
    
    iStart = clock();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    iEnd = clock();
    // TODO : Store time
    
    // TODO : Alloc device global memory
    
    // TODO : Transfer data to device
    
    // TODO : Config and envoke kernel
    
    // TODO : Sum matrix on GPU (TODO : Time it)
    
    // TODO : Copy back computed GPU data
    
    // TODO : Compare host and GPU
    
    // TODO : Host and Device clean up
    
    return (0);
}

