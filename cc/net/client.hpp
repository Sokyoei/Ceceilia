/**
 * @file client.hpp
 * @date 2023/12/13
 * @author Sokyoei
 * @details
 * C++ client
 */

// TCP/IP
// +--------------+
// | Application  |
// +--------------+
// | Presentation |
// +--------------+
// | Session      |
// +--------------+
// | Transport    |
// +--------------+
// | Network      |
// +--------------+
// | Data Link    |
// +--------------+
// | Physical     |
// +--------------+

#pragma once
#ifndef CLIENT_HPP
#define CLIENT_HPP

#include <iostream>
#ifdef _WIN32
// GCC redefine _WIN32_WINNT for link ws2_32
#ifdef __GNUG__
#ifdef _WIN32_WINNT
#undef _WIN32_WINNT
#endif
#define _WIN32_WINNT 0x0600
#endif
#include <WS2tcpip.h>
#include <WinSock2.h>

#pragma comment(lib, "ws2_32.lib")
#elif defined(__linux__)
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#else
#error "This platform Socket operator are not support"
#endif

namespace Ahri {
class Client {
public:
    Client(std::string ip, int port) {
        WSADATA wsadata;
        if (WSAStartup(MAKEWORD(2, 2), &wsadata) == 0) {
            // create socket
            SOCKET clientsock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
            // send infomation
            SOCKADDR_IN sockaddr;
            memset(&sockaddr, 0, sizeof(sockaddr));
            sockaddr.sin_family = PF_INET;  // ipv4
            // #ifdef _MSC_VER
            inet_pton(PF_INET, ip.c_str(), &sockaddr.sin_addr.S_un.S_addr);
            // #elif defined(__GNUC__)
            //         sockaddr.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");
            // #endif
            sockaddr.sin_port = htons(port);
            connect(clientsock, (SOCKADDR*)&sockaddr, sizeof(sockaddr));

            while (true) {
                std::string s;
                char buffer[1024] = {0};

                std::cout << "send info: ";
                std::cin >> s;
                // const char* msg = "hello sokyoei";
                send(clientsock, s.c_str(), s.length(), 0);
                recv(clientsock, buffer, 1024, 0);
                std::cout << "server: " << buffer << std::endl;
            }

            closesocket(clientsock);
            WSACleanup();
        }
    }
    ~Client() {}

private:
    SOCKET client_socket;  // socket
    SOCKADDR_IN sockaddr;  // address
};

void client() {
    WSADATA wsadata;
    if (WSAStartup(MAKEWORD(2, 2), &wsadata) == 0) {
        // create socket
        SOCKET clientsock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
        // send infomation
        SOCKADDR_IN sockaddr;
        memset(&sockaddr, 0, sizeof(sockaddr));
        sockaddr.sin_family = PF_INET;  // ipv4
                                        // #ifdef _MSC_VER
        inet_pton(PF_INET, "127.0.0.1", &sockaddr.sin_addr.S_un.S_addr);
        // #elif defined(__GNUC__)
        //         sockaddr.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");
        // #endif
        sockaddr.sin_port = htons(8888);
        connect(clientsock, (SOCKADDR*)&sockaddr, sizeof(sockaddr));

        while (true) {
            std::string s;
            char buffer[1024] = {0};

            std::cout << "send info: ";
            std::cin >> s;
            // const char* msg = "hello sokyoei";
            send(clientsock, s.c_str(), s.length(), 0);
            recv(clientsock, buffer, 1024, 0);
            std::cout << "server: " << buffer << std::endl;
        }

        closesocket(clientsock);
        WSACleanup();
    }
}
}  // namespace Ahri

#endif  // !CLIENT_HPP
