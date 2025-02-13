/**
 * @file sync.cpp
 * @date 2025/02/08
 * @author Sokyoei
 *
 *
 */

#include <barrier>
#include <chrono>
#include <iostream>
#include <latch>
#include <thread>
#include <vector>

namespace Ahri {
std::latch latch_work(3);
void latch_worker(int id) {
    std::cout << "[" << id << "] start" << '\n';
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "[" << id << "] stop" << '\n';
    latch_work.count_down();
}

void latch_example() {
    std::vector<std::thread> threads;
    for (int i = 0; i < 3; i++) {
        threads.emplace_back(latch_worker, i);
    }
    std::cout << "main thread" << '\n';
    latch_work.wait();
    std::cout << "main thread continue" << '\n';

    for (auto&& t : threads) {
        t.join();
    }
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    Ahri::latch_example();
    return 0;
}
