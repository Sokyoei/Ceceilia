#include <iostream>

#include <jemalloc/jemalloc.h>

int main(int argc, char const* argv[]) {
    // 初始化 jemalloc
    mallctl("version", NULL, NULL, NULL, 0);

    // 分配内存
    size_t size = 1024;
    void* ptr = malloc(size);

    if (ptr != NULL) {
        std::cout << "成功分配了 " << size << " 字节的内存" << '\n';

        // 使用分配的内存
        char* char_ptr = static_cast<char*>(ptr);
        for (size_t i = 0; i < size; ++i) {
            char_ptr[i] = static_cast<char>(i % 256);
        }

        // 获取内存统计信息
        size_t allocated = 0;
        size_t resident = 0;
        size_t active = 0;
        size_t sz = sizeof(size_t);
        mallctl("stats.allocated", &allocated, &sz, NULL, 0);
        mallctl("stats.resident", &resident, &sz, NULL, 0);
        mallctl("stats.active", &active, &sz, NULL, 0);

        std::cout << "已分配内存: " << allocated << " 字节" << '\n';
        std::cout << "常驻内存: " << resident << " 字节" << '\n';
        std::cout << "活跃内存: " << active << " 字节" << '\n';

        // 释放内存
        free(ptr);
        std::cout << "内存已释放" << '\n';
    } else {
        std::cerr << "内存分配失败" << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
