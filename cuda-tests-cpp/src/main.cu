#include <iostream>

__global__ void hello_from_gpu()
{
    printf("Hello from GPU!\n");
}

__global__ void VecAdd(int* A, int* B, int* C)
{
    int i = threadIdx.x;
    printf("idx = %i\n", i);
    C[i] = A[i] + B[i];
}

void printVec(int* A, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        printf("%d ", A[i]);
    }
    printf("\n");
}

int main()
{
    std::cout << "Hello from CPU!" << std::endl;

    size_t size = 10;

    int* A = new int[size];
    int* B = new int[size];
    int* C = new int[size];

    for (size_t i = 0; i < size; i++)
    {
        A[i] = i;
        B[i] = size + i;
        C[i] = 0;
    }

    int* devA;
    cudaMalloc(&devA, sizeof(int) * size);
    int* devB;
    cudaMalloc(&devB, sizeof(int) * size);
    int* devC;
    cudaMalloc(&devC, sizeof(int) * size);

    cudaMemcpy(devA, A, sizeof(int) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, sizeof(int) * size, cudaMemcpyHostToDevice);

    VecAdd<<<1, size>>>(devA, devB, devC);

    cudaDeviceSynchronize();

    cudaMemcpy(C, devC, sizeof(int) * size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    printVec(A, 10);
    printVec(B, 10);
    printVec(C, 10);

    delete[] A;
    delete[] B;
    delete[] C;

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    return 0;
}
