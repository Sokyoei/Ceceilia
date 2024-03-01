/**
 * @file server.hpp
 * @date 2023/12/13
 * @author Sokyoei
 * @details
 * C++ server
 */

#pragma once
#ifndef SERVER_HPP
#define SERVER_HPP

#include <iostream>
#include <string>

#ifdef _WIN32
#ifdef __GNUG__
// #define _WIN32_WINNT 0x0A00  // Windows 10
#endif
#include <WS2tcpip.h>
#include <WinSock2.h>
#include <Windows.h>

#pragma comment(lib, "ws2_32.lib")
#elif defined(__linux__)
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#else
#error "This platform Socket operator are not support"
#endif

namespace Ahri {
class Server {
#ifdef __linux__
    using SOCKET = int;
    using SOCKADDR_IN = sockaddr_in;
#endif
public:
    Server(std::string ip, int port) {
#ifdef _WIN32
        WSADATA wsadata;
        int result = WSAStartup(MAKEWORD(2, 2), &wsadata);
        if (result != 0) {
            switch (result) {
                case WSASYSNOTREADY:
                    break;
                case WSAVERNOTSUPPORTED:
                    break;
                case WSAEINPROGRESS:
                    break;
                case WSAEPROCLIM:
                    break;
                case WSAEFAULT:
                    break;
                default:
                    break;
            }
        }
#endif
        server_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        sockaddr.sin_family = AF_INET;
        inet_pton(PF_INET, ip.c_str(), &sockaddr.sin_addr.S_un.S_addr);
        sockaddr.sin_port = htons(port);
        bind(server_socket, (SOCKADDR*)&sockaddr, sizeof(SOCKADDR));
        listen(server_socket, 20);
        int addrlen = sizeof(SOCKADDR);
        client_socket = accept(server_socket, &clientaddr, &addrlen);
    }

    ~Server() {
#ifdef _WIN32
        closesocket(server_socket);
        closesocket(client_socket);
        WSACleanup();
#elif defined(__linux__)
        close(server_socket);
        close(client_socket);
#endif
    }

private:
    SOCKET server_socket;
    SOCKET client_socket;
    SOCKADDR_IN sockaddr;
    SOCKADDR clientaddr;
};

void server() {
    WSADATA wsadata;
    if (WSAStartup(MAKEWORD(2, 2), &wsadata) == 0) { /* init success */
        // create socket
        SOCKET serversock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
        // bind
        SOCKADDR_IN sockaddr;
        memset(&sockaddr, 0, sizeof(sockaddr));
        sockaddr.sin_family = PF_INET;  // ipv4
        // sockaddr.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");
        sockaddr.sin_addr.s_addr = htonl(INADDR_ANY);
        // inet_pton(PF_INET, "127.0.0.1", &sockaddr.sin_addr.S_un.S_addr);
        // #ifdef _MSC_VER
        //         inet_pton(PF_INET, "127.0.0.1", &sockaddr.sin_addr.S_un.S_addr);
        // #elif defined(__GNUC__)
        //         sockaddr.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");
        // #endif

        sockaddr.sin_port = htons(8888);
        bind(serversock, (SOCKADDR*)&sockaddr, sizeof(SOCKADDR));
        // listen socket
        listen(serversock, 20);
        // link client
        SOCKADDR clientaddr;
        int nsize = sizeof(SOCKADDR);
        SOCKET clientsock = accept(serversock, (SOCKADDR*)&clientaddr, &nsize);
        // send infomation

        while (true) {
            std::string s;
            char buffer[1024] = {0};
            // const char* msg = "hello ahri";
            std::cout << "send info: ";
            std::cin >> s;
            send(clientsock, s.c_str(), s.length(), 0);
            recv(clientsock, buffer, 1024, 0);
            std::cout << "client: " << buffer << std::endl;
        }

        // close sockets
        closesocket(clientsock);
        closesocket(serversock);
        // clean dll
        WSACleanup();
    }
}
}  // namespace Ahri

#endif  // !SERVER_HPP
