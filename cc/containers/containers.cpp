/**
 * @file containers.cpp
 * @date 2023/12/20
 * @author Sokyoei
 * @details
 * C++ containers
 */

#include <array>
#include <deque>
#include <forward_list>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Ahri {
/**
 * @brief Sequence containers(顺序容器)
 */
using std::array;         // 静态连续数组，代替 C 数组
using std::deque;         // 双向动态连续数组
using std::forward_list;  // 单向链表
using std::list;          // 双向链表
using std::vector;        // 动态连续数组

/**
 * @brief Associative containers(关联容器)，底层实现是红黑树
 */
using std::map;       // 不可重复的有序键值对
using std::multimap;  // 可重复的有序键值对
using std::multiset;  // 可重复的有序键集合
using std::set;       // 不可重复的有序键集合

/**
 * @brief Unordered associative containers(无序关联容器)，底层实现是哈希表
 */
using std::unordered_map;       // 不可重复的无序键值对
using std::unordered_multimap;  // 可重复的无序键值对
using std::unordered_multiset;  // 可重复的无序键集合
using std::unordered_set;       // 不可重复的无序键集合

/**
 * @brief Container adaptors(容器适配器)
 */
using std::priority_queue;  // 优先队列
using std::queue;           // 队列，先进先出(FIFO)
using std::stack;           // 堆栈，先进后出(LIFO)
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    return 0;
}
