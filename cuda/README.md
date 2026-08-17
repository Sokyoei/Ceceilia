# CUDA

[docs](https://docs.nvidia.com/cuda/)

## 安装与卸载

安装

=== "Windows"

    下载安装包并运行

=== "Linux"

    !!! warning "linux cudnn 直接解压要解压到 `/usr/local/cuda-X.Y/target/linux.../` 路径下"

        ```shell
        sudo tar -xvf cudnn-linux-x86_64-X.Y.Z_cudaXX-archive.tar.xz -C /usr/local/cuda/targets/x86_64-linux/  --strip-components 1
        ```

卸载

=== "Windows"

    在卸载程序界面卸载

=== "Linux"

    ```shell
    cd /usr/local-X.Y/cuda
    sudo ./cuda-uninstaller
    sudo ./uninstall_cuda_X.Y.pl  # CUDA 10.0 以下
    sudo rm -rf /usr/local/cuda-X.Y
    ```

## nvcc

| 参数             | 描述                   |
| :--------------- | :--------------------- |
| -arch=compute_XY | 指定虚拟架构的计算能力 |
| -code=sm_XY      | 指定真实架构的计算能力 |

## PTX

PTX(Parallel Thread Execution)

```shell
nvcc -ptx cuda_file.cu -o ptx_file.ptx
```

## kernel function 核函数

1. 核函数只能访问 GPU 显存
2. 不能使用变长参数
3. 不能使用静态变量
4. 不能使用函数指针
5. 核函数具有异步性

核函数

```c
__global__ void function_name(args) {
    // TODO
}
```

核函数调用

```c
function_name<<<grid_size, block_size>>>(args);
```
