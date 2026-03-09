#include <cstring>
#include <iostream>

namespace Ahri {
int global_value = 10;                // 全局变量，存储在全局/静态区
static int static_global_value = 20;  // 静态全局变量，存储在全局/静态区

/**
 * @brief 栈内存区
 */
void stack_memory() {
    std::cout << "栈内存区：" << '\n';

    // 局部变量存储在栈上
    int local_var = 100;
    int local_array[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    std::cout << "局部变量 local_var 地址: " << &local_var << ", 值: " << local_var << '\n';
    std::cout << "局部数组 local_array 地址: " << (void*)local_array << ", 内容: " << local_array << '\n';
    // 栈内存在函数调用结束会自动释放
}

/**
 * @brief 堆内存区
 */
void heap_memory() {
    std::cout << "堆内存区：" << '\n';

    // 在堆上分配内存
    int* heap_int = new int(200);
    char* heap_str = new char[20];
    strcpy(heap_str, "heap memory");

    std::cout << "堆上分配的整数地址: " << heap_int << ", 值: " << *heap_int << '\n';
    std::cout << "堆上分配的字符串地址: " << (void*)heap_str << ", 内容: " << heap_str << '\n';

    // 释放堆内存
    delete heap_int;
    delete[] heap_str;
}

/**
 * @brief 全局/静态内存区
 */
void global_static_memory() {
    std::cout << "全局/静态区：" << '\n';

    // 全局变量
    std::cout << "全局变量 global_value 地址: " << &global_value << ", 值: " << global_value << '\n';
    // 静态全局变量
    std::cout << "静态全局变量 static_global 地址: " << &static_global_value << ", 值: " << static_global_value << '\n';

    // 函数内静态变量
    static int static_local = 300;  // 存储在全局/静态区
    std::cout << "静态局部变量 static_local 地址: " << &static_local << ", 值: " << static_local << '\n';
}

/**
 * @brief 常量内存区
 */
void constant_memory() {
    std::cout << "常量区：" << '\n';

    // 字符串字面值存储在常量区
    const char* const_str = "Hello Constant Memory";
    std::cout << "字符串字面值地址: " << (void*)const_str << ", 内容: " << const_str << '\n';

    // 常量变量
    const int const_val = 42;
    std::cout << "常量变量 const_val 地址: " << &const_val << ", 值: " << const_val << '\n';
}

void memory_layout_comparison() {
    std::cout << "内存布局比较：" << '\n';

    int stack_var = 1;
    static int static_var = 2;
    int* heap_var = new int(3);
    const char* const_str = "constant";

    std::cout << "栈变量地址: " << &stack_var << '\n';
    std::cout << "静态变量地址: " << &static_var << '\n';
    std::cout << "堆变量地址: " << heap_var << '\n';
    std::cout << "常量字符串地址: " << (void*)const_str << '\n';

    delete heap_var;
}

class ObjectMemory {
public:
    int member_var;
    static int static_member;  // 静态成员存储在全局/静态区

    ObjectMemory(int val) : member_var(val) {}

    void show_addresses() {
        std::cout << "对象实例地址: " << this << '\n';
        std::cout << "成员变量 member_var 地址: " << &member_var << '\n';
        std::cout << "静态成员变量 static_member 地址: " << &static_member << '\n';
    }
};

int ObjectMemory::static_member = 400;  // 静态成员初始化
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    Ahri::stack_memory();
    Ahri::heap_memory();
    Ahri::global_static_memory();
    Ahri::constant_memory();
    Ahri::memory_layout_comparison();

    Ahri::ObjectMemory obj(500);
    obj.show_addresses();

    return 0;
}
