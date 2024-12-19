#include <iostream>

#include <cuda_runtime.h>

#include "Ahri.cuh"

// CUDA内核函数
__global__ void kernelFunction(int* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] += 1;
}

int main() {
    const int dataSize = 256;
    const int bytes = dataSize * sizeof(int);

    // 分配主机和设备内存
    int* h_data = (int*)malloc(bytes);
    int* d_data;
    cudaMalloc(&d_data, bytes);

    // 初始化主机数据
    for (int i = 0; i < dataSize; i++) {
        h_data[i] = i;
    }

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 异步内存传输：将主机数据传输到设备
    cudaMemcpyAsync(d_data, h_data, bytes, cudaMemcpyHostToDevice, stream);

    // 在流中启动CUDA内核
    kernelFunction<<<dataSize / 256, 256, 0, stream>>>(d_data);

    // 异步内存传输：将结果从设备传输到主机
    cudaMemcpyAsync(h_data, d_data, bytes, cudaMemcpyDeviceToHost, stream);

    // 等待流中的所有操作完成
    cudaStreamSynchronize(stream);

    // 打印结果
    for (int i = 0; i < dataSize; i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    // 释放资源
    cudaStreamDestroy(stream);
    cudaFree(d_data);
    free(h_data);

    return 0;
}
