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
    // std::async 异步调用
    // -----------------------------------------------------------------------------------------------------------------
    // std::async 策略
    // std::launch::async 调用时异步执行
    // std::launch::deferred std::future::get() 调用时再执行
    // -----------------------------------------------------------------------------------------------------------------
    std::future<std::string> result =
        std::async(std::launch::async | std::launch::deferred, search_database, "Sokyoei");
    std::cout << "main thread do something" << std::endl;
    // 获取 std::future 的值
    std::string data = result.get();
    std::cout << data << std::endl;
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    Ahri::test_async();
    return 0;
}
