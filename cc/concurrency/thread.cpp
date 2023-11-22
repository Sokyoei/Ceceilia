#include <barrier>
#include <condition_variable>
#include <future>
#include <iostream>
#include <latch>
#include <mutex>
#include <semaphore>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <thread>

namespace Ahri {
using std::mutex;
using std::recursive_mutex;
using std::recursive_timed_mutex;
using std::shared_mutex;
using std::shared_timed_mutex;
using std::timed_mutex;

using std::lock;
using std::lock_guard;
using std::scoped_lock;
using std::shared_lock;
using std::unique_lock;

using std::barrier;
using std::latch;

using std::binary_semaphore;
using std::counting_semaphore;

using std::condition_variable;
using std::condition_variable_any;

using std::async;
using std::future;
using std::launch;
using std::packaged_task;
using std::promise;
using std::shared_future;

using std::jthread;
using std::thread;

void say_hello(std::string_view& str) {
    std::cout << "ref: hello " << str << std::endl;
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    std::string_view a = "asd";
    std::thread t(Ahri::say_hello, std::ref(a));
    t.join();
    return 0;
}
