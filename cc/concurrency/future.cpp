/**
 * @file future.cpp
 * @date 2024/03/22
 * @author Sokyoei
 * @details
 *
 */

#include <chrono>
#include <future>
#include <iostream>
#include <thread>

namespace Ahri {
std::string search_database(std::string query) {
    std::this_thread::sleep_for(std::chrono::seconds(5));
    return "data: " + query;
}

void test_async() {
    // std::async 策略
    // std::launch::async 调用时异步执行
    // std::launch::deferred std::future::get() 调用时再执行
    std::future<std::string> result =
        std::async(std::launch::async | std::launch::deferred, search_database, "Sokyoei");
    std::cout << "main thread do something" << std::endl;
    // 获取 std::future 的值
    std::string data = result.get();
    std::cout << data << std::endl;
}

void test_package_task() {
    std::packaged_task<std::string(std::string)> task(search_database);
    std::future<std::string> result = task.get_future();
    // 开启一个线程执行任务
    std::thread t(std::move(task), "Sokyoei");
    // 与主线程分离，主线程可以等待任务完成
    t.detach();

    std::cout << "value: " << result.get() << std::endl;
}

void search_string(std::promise<std::string> prom) {
    std::this_thread::sleep_for(std::chrono::seconds(5));
    prom.set_value("Sokyoei");
    std::cout << "subthread search_string end" << std::endl;
}

void test_promise() {
    std::promise<std::string> prom;
    std::future<std::string> fut = prom.get_future();
    std::thread t(search_string, std::move(prom));
    std::cout << "waiting for subthread to set value" << std::endl;
    std::cout << "value: " << fut.get() << std::endl;
    t.join();
}

void set_exception(std::promise<void> prom) {
    try {
        throw std::runtime_error("something wrong");
    } catch (...) {
        prom.set_exception(std::current_exception());
    }
}

void test_promise_exception() {
    std::promise<void> prom;
    std::future<void> fut = prom.get_future();
    std::thread t(set_exception, std::move(prom));
    try {
        std::cout << "waiting for subthread to set exception" << std::endl;
        fut.get();
    } catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << '\n';
    }
    t.join();
}

void shared_subthread(std::shared_future<std::string> fut) {
    try {
        std::string result = fut.get();
        std::cout << "result: " << result << std::endl;
    } catch (const std::future_error& e) {
        std::cout << "future error: " << e.what() << '\n';
    }
}

void test_shared_future() {
    std::promise<std::string> prom;
    // 隐式转换 std::future -> std::shared_future
    std::shared_future<std::string> shared_fut = prom.get_future();

    std::thread t1(search_string, std::move(prom));
    std::thread t2(shared_subthread, shared_fut);
    std::thread t3(shared_subthread, shared_fut);
    t1.join();
    t2.join();
    t3.join();
}

void throw_exception() {
    throw std::runtime_error("something wrong");
}

void test_future_exception() {
    std::future<void> result(std::async(std::launch::async, throw_exception));
    try {
        result.get();
    } catch (const std::exception& e) {
        std::cout << "catch error: " << e.what() << '\n';
    }
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    Ahri::test_async();
    Ahri::test_package_task();
    Ahri::test_promise();
    Ahri::test_promise_exception();
    Ahri::test_shared_future();
    Ahri::test_future_exception();
    return 0;
}
