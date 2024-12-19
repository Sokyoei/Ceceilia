#include <cuda_runtime.h>
#include <iostream>

// 简单的CUDA内核函数
__global__ void simpleKernel(int* data, int value) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = value;
}

int main() {
    const int dataSize = 256;
    const int bytes = dataSize * sizeof(int);

    // 分配主机和设备内存
    int* h_data = (int*)malloc(bytes);
    int* d_data;
    cudaMalloc(&d_data, bytes);

    // 创建CUDA事件
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // 记录开始事件
    cudaEventRecord(startEvent, 0);

    // 启动CUDA内核
    simpleKernel<<<dataSize / 256, 256>>>(d_data, 42);

    // 记录结束事件
    cudaEventRecord(stopEvent, 0);

    // 等待事件完成
    cudaEventSynchronize(stopEvent);

    // 计算内核执行时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);

    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // 释放资源
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_data);
    free(h_data);

    return 0;
}
