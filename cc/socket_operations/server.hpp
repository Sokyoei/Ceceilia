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
class Server {
private:
public:
    Server() {}
    ~Server() {}
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
        inet_pton(PF_INET, "127.0.0.1", &sockaddr.sin_addr.S_un.S_addr);
        sockaddr.sin_port = htons(8888);
        bind(serversock, (SOCKADDR*)&sockaddr, sizeof(SOCKADDR));
        // listen socket
        listen(serversock, 20);
        // link client
        SOCKADDR clientaddr;
        int nsize = sizeof(SOCKADDR);
        SOCKET clientsock = accept(serversock, (SOCKADDR*)&clientaddr, &nsize);
        // send infomation
        const char* msg = "hello ahri";
        send(clientsock, msg, strlen(msg) + sizeof(char), 0);

        char buffer[1024] = {0};
        recv(clientsock, buffer, 1024, 0);
        std::cout << "client: " << buffer << std::endl;
        // close sockets
        closesocket(clientsock);
        closesocket(serversock);
        // clean dll
        WSACleanup();
    }
}
}  // namespace Ahri
