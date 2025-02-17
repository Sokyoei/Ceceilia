#include <chrono>
#include <thread>

// for VSCode IntelliSense
#ifndef GLOG_USE_GLOG_EXPORT
#define GLOG_USE_GLOG_EXPORT
#endif

#include <fmt/std.h>
#include <folly/String.h>
#include <folly/executors/CPUThreadPoolExecutor.h>

int main(int argc, char const* argv[]) {
    folly::CPUThreadPoolExecutor executor(4);

    auto task = []() {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        fmt::println("Task executed in thread: {}", std::this_thread::get_id());
    };

    for (int i = 0; i < 10; ++i) {
        executor.add(task);
    }

    executor.join();

    return 0;
}
