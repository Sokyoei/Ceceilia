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
#include <string>

#include <boost/asio.hpp>

namespace Ahri {
class Server {
public:
    Server(std::string ip, int port) {
        socket.open(boost::asio::ip::tcp::v4());
        socket.bind(boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port), error_code);
    }
    ~Server() { socket.close(error_code); }

private:
    boost::asio::io_context io;  // io 上下文对象
    boost::asio::ip::tcp::socket socket{io};
    boost::system::error_code error_code;
};
}  // namespace Ahri

#endif  // !TCP_SERVER_HPP
