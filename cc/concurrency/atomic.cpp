/**
 * @file atomic.cpp
 * @date 2023/12/22
 * @author Sokyoei
 * @details
 * C++ atomic 原子操作
 */

#include "Ahri/Ahri.hpp"

#include <atomic>

namespace Ahri {
/**
 * @details 内存次序
 * 宽松内存序 @c std::memory_order_relaxed
 *
 */

///
/// @details 原子类型
/// +------------------------------+-----------------------------------------------------------------------------------+
/// | @b std::memory_order         | @b 说明
/// | @c std::memory_order_seq_cst | 顺序一致性，最强内存顺序，默认选项
/// | @c std::memory_order_acq_rel | 获取-释放操作，用于读取数据和写入数据，兼具 acquire 和 release 语义
/// | @c std::memory_order_acquire | 获取操作，用于读取数据
/// | @c std::memory_order_consume | 消费操作
/// | @c std::memory_order_relaxed | 最宽松，只保证原子性，不保证顺序
/// | @c std::memory_order_release | 释放操作，用于写入数据
/// +------------------------------+-----------------------------------------------------------------------------------+
///

void atomic_example() {
    std::atomic_int counter{0};
    int value = counter.load(std::memory_order_relaxed);
    counter.store(13, std::memory_order_relaxed);
    int old_value = counter.exchange(14, std::memory_order_relaxed);
}

void atomic_flag_exampe() {
    std::atomic_flag flag = ATOMIC_FLAG_INIT;
}

#ifdef AHRI_CXX20
void atomic_ref_example() {
    std::atomic_ref<int> ref{1};
}
#endif
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    Ahri::atomic_flag_exampe();
#ifdef AHRI_CXX20
    Ahri::atomic_ref_example();
#endif
    return 0;
}
