/**
 * @file Ahri.cpp
 * @date 2024/02/06
 * @author Sokyoei
 *
 *
 */

#include <iostream>

#include <fmt/core.h>
#include <boost/asio.hpp>

#ifndef _WIN32_WINNT
#define _WIN32_WINNT = 0x0A00  // Windows 10
#endif

namespace Ahri {
void handler(const boost::system::error_code& ec) {
    fmt::println("5 s.");
}

void print(const boost::system::error_code& ec, boost::asio::steady_timer* t, int* count) {
    if (*count < 5) {
        fmt::println("{}", *count);
        ++(*count);

        t->expires_at(t->expiry() + boost::asio::chrono::seconds(1));
        t->async_wait(std::bind(print, boost::asio::placeholders::error, t, count));
    }
}

class printer {
public:
    printer(boost::asio::io_context& io) : _timer(io, boost::asio::chrono::seconds(1)), _count(0) {
        _timer.async_wait(std::bind(&printer::print, this));
    }

    ~printer() { std::cout << "Final count is " << _count << std::endl; }

    void print() {
        if (_count < 5) {
            fmt::println("{}", _count);
            ++_count;

            _timer.expires_at(_timer.expiry() + boost::asio::chrono::seconds(1));
            _timer.async_wait(std::bind(&printer::print, this));
        }
    }

private:
    boost::asio::steady_timer _timer;
    int _count;
};
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    boost::asio::io_context io_context;

    // sync
    boost::asio::steady_timer stimer(io_context, boost::asio::chrono::seconds(5));
    stimer.wait();
    fmt::println("main Hello, world!");

    // async
    boost::asio::deadline_timer timer(io_context, boost::posix_time::seconds(5));
    timer.async_wait(Ahri::handler);

    int count = 0;
    boost::asio::steady_timer t(io_context, boost::asio::chrono::seconds(1));
    t.async_wait(std::bind(Ahri::print, boost::asio::placeholders::error, &t, &count));

    Ahri::printer p(io_context);
    io_context.run();

    fmt::println("Final count is {}", count);

    return 0;
}
