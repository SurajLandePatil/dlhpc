#include <iostream>
#include <cuda_runtime.h>

#define N 1000000       // Size of vector
#define M 512           // Size of square matrices for multiplication

// CUDA kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

// CUDA kernel for matrix multiplication
__global__ void matrixMultiply(float *A, float *B, float *C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        float value = 0;
        for (int k = 0; k < size; ++k) {
            value += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = value;
    }
}

void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // ---------------------- Vector Addition --------------------------
    float *h_A = new float[N], *h_B = new float[N], *h_C = new float[N];
    for (int i = 0; i < N; ++i) {
        h_A[i] = rand() % 100;
        h_B[i] = rand() % 100;
    }

    float *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc((void**)&d_A, N * sizeof(float)), "Failed to allocate d_A");
    checkCuda(cudaMalloc((void**)&d_B, N * sizeof(float)), "Failed to allocate d_B");
    checkCuda(cudaMalloc((void**)&d_C, N * sizeof(float)), "Failed to allocate d_C");

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Vector Addition Done.\n";

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;

    // ---------------------- Matrix Multiplication --------------------------
    int size = M * M;
    float *h_MA = new float[size];
    float *h_MB = new float[size];
    float *h_MC = new float[size];

    for (int i = 0; i < size; ++i) {
        h_MA[i] = rand() % 100;
        h_MB[i] = rand() % 100;
    }

    float *d_MA, *d_MB, *d_MC;
    checkCuda(cudaMalloc((void**)&d_MA, size * sizeof(float)), "Failed to allocate d_MA");
    checkCuda(cudaMalloc((void**)&d_MB, size * sizeof(float)), "Failed to allocate d_MB");
    checkCuda(cudaMalloc((void**)&d_MC, size * sizeof(float)), "Failed to allocate d_MC");

    cudaMemcpy(d_MA, h_MA, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_MB, h_MB, size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock2D(16, 16);
    dim3 blocksPerGrid2D((M + 15) / 16, (M + 15) / 16);

    matrixMultiply<<<blocksPerGrid2D, threadsPerBlock2D>>>(d_MA, d_MB, d_MC, M);

    cudaMemcpy(h_MC, d_MC, size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Matrix Multiplication Done.\n";

    cudaFree(d_MA); cudaFree(d_MB); cudaFree(d_MC);
    delete[] h_MA; delete[] h_MB; delete[] h_MC;

    return 0;
}
