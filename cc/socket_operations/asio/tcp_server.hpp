/**
 * @file tcp_server.hpp
 * @date 2024/02/05
 * @author Sokyoei
 *
 *
 */

#pragma once
#ifndef TCP_SERVER_HPP
#define TCP_SERVER_HPP

#include <iostream>

#include <boost/asio.hpp>

namespace Ahri {
class Server {
private:
public:
    Server() {
        boost::asio::io_context io;  // io 上下文对象
        boost::asio::ip::tcp::socket socket(io);
        socket.open(boost::asio::ip::tcp::v4());
        boost::asio::error::basic_errors be;
        auto error_code = boost::asio::error::make_error_code(be);
        socket.bind(boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), 12345), error_code);

        socket.close(error_code);

        boost::asio::ip::tcp::acceptor;
        boost::asio::ip::udp::socket;
        boost::asio::deadline_timer;
    }
    ~Server() {}
};
}  // namespace Ahri

#endif  // !TCP_SERVER_HPP
