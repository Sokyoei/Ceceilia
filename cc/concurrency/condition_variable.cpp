#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

namespace Ahri {
void test_condition_variable() {
    int num = 1;
    std::mutex mutex;
    std::condition_variable cv1;
    std::condition_variable cv2;

    std::thread t1([&]() {
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            // 写法一
            // while (num != 1) {
            //     cv2.wait(lock);
            // }
            // 写法二
            cv1.wait(lock, [&]() { return num == 1; });
            num++;
            std::cout << "thread " << std::this_thread::get_id() << ": " << num << std::endl;
            cv2.notify_one();
        }
    });
    std::thread t2([&]() {
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            cv2.wait(lock, [&]() { return num == 2; });
            num--;
            std::cout << "thread " << std::this_thread::get_id() << ": " << num << std::endl;
            cv1.notify_one();
        }
    });

    t1.join();
    t2.join();
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    Ahri::test_condition_variable();
    return 0;
}
