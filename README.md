# CUDA high peformance computing

#include <stdio.h>

#define N 256
#define BIN_COUNT 16

__global__ void histogram(int* input, int* bins, int binCount) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        int value = input[tid];
        int binIndex = value % binCount;  // Example binning logic
        atomicAdd(&bins[binIndex], 1);   // Increment bin atomically
    }
}

int main() {
    int input[N], bins[BIN_COUNT] = {0};
    int *d_input, *d_bins;

    // Initialize input array with random values
    for (int i = 0; i < N; i++) {
        input[i] = i % 16;  // Example values between 0 and 15
    }

    // Allocate device memory
    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_bins, BIN_COUNT * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_input, input, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bins, bins, BIN_COUNT * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 128;
    int numBlocks = (N + blockSize - 1) / blockSize;
    histogram<<<numBlocks, blockSize>>>(d_input, d_bins, BIN_COUNT);

    // Copy result back to host
    cudaMemcpy(bins, d_bins, BIN_COUNT * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    printf("Histogram:\n");
    for (int i = 0; i < BIN_COUNT; i++) {
        printf("Bin %d: %d\n", i, bins[i]);
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_bins);

    return 0;
}
 
2d-grid image processing:

#include <stdio.h>

#define WIDTH 16
#define HEIGHT 16

__global__ void invertImage(unsigned char* input, unsigned char* output, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        output[idx] = 255 - input[idx];  // Invert pixel value
    }
}

int main() {
    unsigned char image[WIDTH * HEIGHT], result[WIDTH * HEIGHT];
    unsigned char *d_image, *d_result;

    // Initialize image with some values
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        image[i] = i % 256;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_image, WIDTH * HEIGHT * sizeof(unsigned char));
    cudaMalloc((void**)&d_result, WIDTH * HEIGHT * sizeof(unsigned char));

    // Copy data to device
    cudaMemcpy(d_image, image, WIDTH * HEIGHT * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(8, 8);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    invertImage<<<gridSize, blockSize>>>(d_image, d_result, WIDTH, HEIGHT);

    // Copy result back to host
    cudaMemcpy(result, d_result, WIDTH * HEIGHT * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Print a small portion of the result
    printf("Inverted image:\n");
    for (int i = 0; i < 16; i++) {
        printf("%d ", result[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_image);
    cudaFree(d_result);

    return 0;
}

Prefix sum array:
#include <stdio.h>

#define N 16

__global__ void prefixSum(int* input, int* output) {
    __shared__ int temp[N];
    int tid = threadIdx.x;

    // Load input into shared memory
    temp[tid] = input[tid];
    __syncthreads();

    // Perform prefix sum
    for (int stride = 1; stride < N; stride *= 2) {
        int val = 0;
        if (tid >= stride) {
            val = temp[tid - stride];
        }
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    // Write output
    output[tid] = temp[tid];
}

int main() {
    int input[N], output[N];
    int *d_input, *d_output;

    // Initialize input array
    for (int i = 0; i < N; i++) {
        input[i] = 1;  // Example input: all elements set to 1
    }

    // Allocate device memory
    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_output, N * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_input, input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    prefixSum<<<1, N>>>(d_input, d_output);

    // Copy result back to host
    cudaMemcpy(output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    printf("Prefix Sum:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", output[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

Matrix-multiplication:

#include <stdio.h>

#define N 3  // Size of matrix (N x N)

__global__ void matrixMul(int* A, int* B, int* C) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int A[N][N] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int B[N][N] = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
    int C[N][N] = {0};

    int *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc((void**)&d_A, N * N * sizeof(int));
    cudaMalloc((void**)&d_B, N * N * sizeof(int));
    cudaMalloc((void**)&d_C, N * N * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockSize(N, N);
    matrixMul<<<1, blockSize>>>(d_A, d_B, d_C);

    // Copy result back to host
    cudaMemcpy(C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    printf("Result matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", C[i][j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

Dot-product:
#include <stdio.h>

#define N 256
#define BLOCK_SIZE 128

__global__ void reduceSum(int* input, int* output, int n) {
    __shared__ int sharedData[BLOCK_SIZE];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tx = threadIdx.x;

    // Load data into shared memory
    sharedData[tx] = (tid < n) ? input[tid] : 0;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tx < s) {
            sharedData[tx] += sharedData[tx + s];
        }
        __syncthreads();
    }

    // Write the result of this block to global memory
    if (tx == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

int main() {
    int A[N], partialSums[(N + BLOCK_SIZE - 1) / BLOCK_SIZE];
    int *d_A, *d_partialSums;

    // Initialize array
    for (int i = 0; i < N; i++) {
        A[i] = i + 1;  // Array: 1, 2, 3, ..., N
    }

    // Allocate device memory
    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_partialSums, sizeof(partialSums));

    // Copy data to device
    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduceSum<<<numBlocks, BLOCK_SIZE>>>(d_A, d_partialSums, N);

    // Copy result back to host
    cudaMemcpy(partialSums, d_partialSums, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    // Final sum reduction on the host
    int totalSum = 0;
    for (int i = 0; i < numBlocks; i++) {
        totalSum += partialSums[i];
    }

    printf("Sum of array: %d\n", totalSum);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_partialSums);

    return 0;
}


Vector add:
#include <stdio.h>

#define N 256  // Size of vectors

__global__ void vectorAdd(int* A, int* B, int* C) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;  // Calculate global thread index
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int A[N], B[N], C[N];
    int *d_A, *d_B, *d_C;

    // Initialize vectors
    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = 2 * i;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_B, N * sizeof(int));
    cudaMalloc((void**)&d_C, N * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

    // Copy result back to host
    cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    printf("Result:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", C[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

 
