/**
 * @file Ahri.cpp
 * @date 2024/02/06
 * @author Sokyoei
 *
 *
 */

#include <iostream>

#include <boost/asio.hpp>

// #define _WIN32_WINNT = 0x0601

void handler(const boost::system::error_code& ec) {
    std::cout << "5 s." << std::endl;
}

int main(int argc, char const* argv[]) {
    boost::asio::io_context io_context;
    boost::asio::deadline_timer timer(io_context, boost::posix_time::seconds(5));
    timer.async_wait(handler);
    io_context.run();
    return 0;
}
