/**
 * @file threadpool_example.cpp
 * @date 2024/03/22
 * @author Sokyoei
 *
 *
 */

#include <chrono>
#include <future>
#include <iostream>
#include <vector>

#include "threadpool.hpp"

int main(int argc, char const* argv[]) {
    ThreadPool pool(4);
    std::vector<std::future<int>> results;
    for (int i = 0; i < 8; ++i) {
        results.emplace_back(pool.enqueue([i] {
            std::cout << "hello" << i << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::cout << "world" << i << std::endl;
            return i * i;
        }));
    }

    for (auto&& result : results) {
        std::cout << result.get() << ' ';
    }
    std::cout << std::endl;

    return 0;
}
