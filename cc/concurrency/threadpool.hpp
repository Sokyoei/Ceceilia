#pragma once
#ifndef THREADPOOL_HPP
#define THREADPOOL_HPP

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

class ThreadPool {
    using Task = std::function<void()>;

private:
    std::condition_variable _cond;
    std::vector<std::thread> _workers;
    std::queue<Task> _tasks;
    std::mutex _mutex;
    bool _stop;

public:
    explicit ThreadPool(size_t threads) : _stop(false) {
        for (size_t i = 0; i < threads; i++) {
            _workers.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->_mutex);
                        this->_cond.wait(lock, [this] { return this->_stop || !this->_tasks.empty(); });
                        if (this->_stop && this->_tasks.empty()) {
                            return;
                        }
                        task = std::move(this->_tasks.front());
                        this->_tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(_mutex);
            _stop = true;
        }
        _cond.notify_all();
        for (auto&& worker : _workers) {
            worker.join();
        }
    }

    template <typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(_mutex);
            if (_stop) {
                throw std::runtime_error("enqueue on stoped ThreadPool");
            }
            _tasks.emplace([task]() { (*task)(); });
        }
        _cond.notify_one();
        return res;
    }
};

#endif  // THREADPOOL_HPP
