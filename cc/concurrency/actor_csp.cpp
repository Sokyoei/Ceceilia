/**
 * @file actor_csp.cpp
 * @date 2024/03/21
 * @author Sokyoei
 * @details
 * Actor CSP
 */

#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

namespace Ahri {
template <typename T>
class Channel {
private:
    std::queue<T> _queue;
    std::mutex _mutex;
    std::condition_variable _cv_producer;
    std::condition_variable _cv_consumer;
    size_t _capacity;
    bool _closed = false;

public:
    Channel(size_t capacity = 0) : _capacity(capacity){};

    bool send(T value) {
        std::unique_lock<std::mutex> lock(_mutex);
        _cv_producer.wait(
            lock, [this]() { return (_capacity == 0 && _queue.empty()) || _queue.size() < _capacity || _closed; });

        if (_closed) {
            return false;
        }

        _queue.push(value);
        _cv_producer.notify_one();
        return true;
    }

    bool recv(T& value) {
        std::unique_lock<std::mutex> lock(_mutex);
        _cv_consumer.wait(lock, [this]() { return !_queue.empty() || _closed; });

        if (_closed && _queue.empty()) {
            return false;
        }

        value = _queue.front();
        _queue.pop();
        _cv_producer.notify_one();
        return true;
    }

    void close() {
        std::unique_lock<std::mutex> lock(_mutex);
        _closed = true;
        _cv_producer.notify_all();
        _cv_consumer.notify_all();
    }
};

void test_channel() {
    Channel<int> ch(10);

    std::thread producer([&]() {
        for (int i = 0; i < 5; i++) {
            ch.send(i);
            std::cout << "send: " << i << std::endl;
        }
        ch.close();
    });

    std::thread consumer([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        int val;
        while (ch.recv(val)) {
            std::cout << "recv: " << val << std::endl;
        }
    });

    producer.join();
    consumer.join();
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    Ahri::test_channel();
    return 0;
}
