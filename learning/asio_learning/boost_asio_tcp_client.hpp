/**
 * @file tcp_client.hpp
 * @date 2024/02/05
 * @author Sokyoei
 *
 *
 */

#pragma once
#ifndef TCP_CLIENT_HPP
#define TCP_CLIENT_HPP

#include <iostream>

#include <boost/asio.hpp>

namespace Ahri {
class Client {
private:
public:
public:
    Client(std::string ip, int port,std::string server_ip) {
        socket.open(boost::asio::ip::tcp::v4());
    }
    ~Client() { socket.close(error_code); }

private:
    boost::asio::io_context io;  // io 上下文对象
    boost::asio::ip::tcp::socket socket{io};
    boost::system::error_code error_code;
};
}  // namespace Ahri

#endif  // !TCP_CLIENT_HPP
