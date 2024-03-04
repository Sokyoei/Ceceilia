/**
 * @file lock.cpp
 * @date 2024/02/26
 * @author Sokyoei
 *
 *
 */

#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stack>
#include <thread>

namespace Ahri {
std::mutex mutex;

void lock() {
    mutex.lock();
    // do something
    mutex.unlock();
}

void lock_raii() {
    std::lock_guard<std::mutex> lock(mutex);
    // do something
}

class empty_stack : public std::exception {
private:
    const char* message;

public:
    const char* what() const throw() { return message; };
};

template <typename T>
class ThreadSafeStack {
private:
    std::stack<T> data;
    mutable std::mutex mutex;

public:
    ThreadSafeStack() {}

    ThreadSafeStack(const ThreadSafeStack& other) {
        std::lock_guard<std::mutex> lock(other.mutex);
        data = other.data;
    }

    ~ThreadSafeStack() {}

    ThreadSafeStack& operator=(const ThreadSafeStack&) = delete;

    void push(T value) {
        std::lock_guard<std::mutex> lock(mutex);
        data.push(std::move(value));
    }

    // BUG
    // T pop() {
    //     std::lock_guard<std::mutex> lock(mutex);
    //     auto element = data.top();
    //     data.pop();
    //     return element;
    // }

    std::shared_ptr<T> pop() {
        std::lock_guard<std::mutex> lock(mutex);
        if (data.empty()) {
            return nullptr;
        }
        std::shared_ptr<T> const res = std::make_shared<T>(data.top());
        data.pop();
        return res;
    }

    void pop(T& value) {
        std::lock_guard<std::mutex> lock(mutex);
        if (data.empty()) {
            throw empty_stack();
        }
        value = data.top();
        data.pop();
    }

    // 危险
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex);
        return data.empty();
    }
};

/**
 * @brief 层级锁
 */
class HierarchicalMutex {
private:
    std::mutex internal_mutex;
    unsigned long const hierarchy_value;                            // 当前层级值
    unsigned long previous_hierarchy_value;                         // 上一层层级值
    static thread_local unsigned long this_thread_hierarchy_value;  // 本线程记录的层级值

public:
    explicit HierarchicalMutex(unsigned long value) : hierarchy_value(value), previous_hierarchy_value(0) {}
    ~HierarchicalMutex() {}
    HierarchicalMutex(const HierarchicalMutex&) = delete;
    HierarchicalMutex& operator=(const HierarchicalMutex&) = delete;

    void lock() {
        check_for_hierarchy_violation();
        internal_mutex.lock();
        update_hierarchy_violation();
    }

    void unlock() {
        if (this_thread_hierarchy_value != hierarchy_value) {
            throw std::logic_error("mutex hierarchy violated");
        }
        this_thread_hierarchy_value = previous_hierarchy_value;
        internal_mutex.unlock();
    }

    bool try_lock() {
        check_for_hierarchy_violation();
        if (internal_mutex.try_lock()) {
            return false;
        }
        update_hierarchy_violation();
        return true;
    }

private:
    void check_for_hierarchy_violation() {
        if (this_thread_hierarchy_value <= hierarchy_value) {
            throw std::logic_error("mutex hierarchy violated");
        }
    }

    void update_hierarchy_violation() {
        previous_hierarchy_value = this_thread_hierarchy_value;
        this_thread_hierarchy_value = hierarchy_value;
    }
};

thread_local unsigned long HierarchicalMutex::this_thread_hierarchy_value(ULONG_MAX);

void test_hierarchy_lock() {
    HierarchicalMutex hmtx1(1000);
    HierarchicalMutex hmtx2(500);

    std::thread t1([&hmtx1, &hmtx2]() {
        hmtx1.lock();
        hmtx2.lock();
        hmtx1.unlock();
        hmtx2.unlock();
    });
    std::thread t2([&hmtx1, &hmtx2]() {
        hmtx2.lock();
        hmtx1.lock();
        hmtx2.unlock();
        hmtx1.unlock();
    });
}

// std::unique 比 std::lock_guard 拥有更多的功能
void use_unique_lock() {
    std::unique_lock<std::mutex> lock(mutex);
    // std::unique_lock<std::mutex> lock(mutex, std::defer_lock);  // 延迟锁
    // std::unique_lock<std::mutex> lock(mutex, std::adopt_lock);  // 领养锁
    // do something
    if (lock.owns_lock()) {
        std::cout << "owns lock" << std::endl;
    }
}

// 共享锁
std::shared_mutex shared_mutex;
std::shared_lock<std::shared_mutex> shared_lock(shared_mutex);
// 递归锁
std::recursive_mutex recursive_mutex;
std::lock_guard<std::recursive_mutex> recursive_lock{recursive_mutex};
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    // Ahri::test_hierarchy_lock();
    return 0;
}
