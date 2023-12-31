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

#include <iostream>

#ifdef _WIN32
#ifdef __GNUG__
#define _WIN32_WINNT 0x0A00  // Windows 10
#endif
#include <WS2tcpip.h>
#include <WinSock2.h>

#pragma comment(lib, "ws2_32.lib")
#elif defined(__linux__)

#else
#error "This platform Socket operator are not support"
#endif

namespace Ahri {
class Client {
private:
public:
    Client() {}
    ~Client() {}
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
        // sockaddr.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");
        inet_pton(PF_INET, "127.0.0.1", &sockaddr.sin_addr.S_un.S_addr);
        sockaddr.sin_port = htons(8888);
        connect(clientsock, (SOCKADDR*)&sockaddr, sizeof(sockaddr));

        char buffer[1024] = {0};
        recv(clientsock, buffer, 1024, 0);
        std::cout << "server: " << buffer << std::endl;

        const char* msg = "hello sokyoei";
        send(clientsock, msg, strlen(msg) + sizeof(char), 0);

        closesocket(clientsock);
        WSACleanup();
    }
}
}  // namespace Ahri
