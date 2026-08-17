#include <chrono>
#include <iostream>
#include <thread>

#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>

using std::chrono_literals::operator""s;

int main(int argc, char const* argv[]) {
    boost::asio::thread_pool pool(4);

    for (int i = 0; i < 10; i++) {
        boost::asio::post(pool, []() {
            std::this_thread::sleep_for(1s);
            std::cout << std::this_thread::get_id() << '\n';
        });
    }

    pool.join();
    return 0;
}
