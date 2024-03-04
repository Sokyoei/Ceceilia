/**
 * @file threadsafe_queue.hpp
 * @date 2024/03/04
 * @author Sokyoei
 *
 *
 */

#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

namespace Ahri {
template <typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue(){};
    ThreadSafeQueue(const ThreadSafeQueue& other) {
        std::lock_guard<std::mutex> lock(other.mutex);
        queue = other.queue;
    }

    void push(T value) {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push(value);
        cond.notify_one();
    }

    void pop(T& value) {
        std::unique_lock<std::mutex> lock(mutex);
        cond.wait(lock, [this]() { return !queue.empty(); });
        value = queue.front();
        queue.pop();
    }

    std::shared_ptr<T> pop() {
        std::unique_lock<std::mutex> lock(mutex);
        cond.wait(lock, [this]() { return !queue.empty(); });
        std::shared_ptr<T> result(std::make_shared<T>(queue.front()));
        queue.pop();
        return result;
    }

    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lock(mutex);
        if (queue.empty()) {
            return false;
        }
        value = queue.front();
        queue.pop();
        return true;
    }

    std::shared_ptr<T> try_pop() {
        std::lock_guard<std::mutex> lock(mutex);
        if (queue.empty()) {
            return std::shared_ptr<T>();
        }
        std::shared_ptr<T> result(std::make_shared<T>(queue.front()));
        queue.pop();
        return result;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.empty();
    }

private:
    std::queue<T> queue;
    mutable std::mutex mutex;
    std::condition_variable cond;
};
}  // namespace Ahri
