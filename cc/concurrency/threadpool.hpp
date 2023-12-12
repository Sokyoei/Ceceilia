/**
 * @file threadpool.hpp
 * @date 2023/12/12
 * @author Sokyoei
 * @details
 * C++ ThreadPool
 * support C++11 ~ C++20
 */

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
#include <type_traits>
#include <vector>

class ThreadPool {
    using Task = std::function<void()>;

private:
    std::condition_variable cond;
    std::vector<std::thread> workers;
    std::queue<Task> tasks;
    std::mutex mutex;
    bool stop;

public:
    explicit ThreadPool(size_t threads) : stop(false) {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->mutex);
                        this->cond.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty()) {
                            return;
                        }
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(mutex);
            stop = true;
        }
        cond.notify_all();
        for (auto&& worker : workers) {
            worker.join();
        }
    }

    template <typename F, typename... Args>
#if __cplusplus >= 201402L || _MSVC_LANG >= 201402L
    decltype(auto) enqueue(F&& f, Args&&... args)
#else
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>
#endif
    {
#if __cplusplus >= 201703L || _MSVC_LANG >= 201703L
        using return_type = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>;
#else
        using return_type = typename std::result_of<F(Args...)>::type;
#endif
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        // auto task = std::make_shared<std::packaged_task<return_type()>>([func = std::forward<F>(f)] { return func();
        // });
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(mutex);
            if (stop) {
                throw std::runtime_error("enqueue on stoped ThreadPool");
            }
            tasks.emplace([task]() { (*task)(); });
        }
        cond.notify_one();
        return res;
    }
};

#endif  // THREADPOOL_HPP
