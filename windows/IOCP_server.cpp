#include <iostream>
#include <vector>

#define WIN32_LEAN_AND_MEAN

// clang-format off
#include <WinSock2.h>
#include <MSWSock.h>
#include <Windows.h>
// clang-format on

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "mswsock.lib")

namespace Ahri {
constexpr static size_t MAXBUFFERSIZE = 1024 * 8;

enum class IOType { Read, Write };

// 自定义结构体扩展OVERLAPPED，用于跟踪I/O操作
typedef struct PER_IO_DATA {
    OVERLAPPED overlapped;
    SOCKET socket;
    WSABUF wsaBuf;
    CHAR buffer[MAXBUFFERSIZE];
    DWORD bytesTransferred;
    DWORD operationType;  // 0-接收 1-发送
} PER_IO_DATA;

// 线程函数处理完成端口事件
DWORD WINAPI WorkerThread(LPVOID lpParam) {
    HANDLE hIOCP = (HANDLE)lpParam;
    DWORD bytesTransferred;
    ULONG_PTR completionKey;
    PER_IO_DATA* perIoData = NULL;

    while (TRUE) {
        // 等待I/O完成事件
        BOOL status =
            GetQueuedCompletionStatus(hIOCP, &bytesTransferred, &completionKey, (LPOVERLAPPED*)&perIoData, INFINITE);

        if (!status) {
            if (perIoData) {
                closesocket(perIoData->socket);
                free(perIoData);
            }
            continue;
        }

        if (bytesTransferred == 0) {
            closesocket(perIoData->socket);
            free(perIoData);
            continue;
        }

        // 根据操作类型处理
        if (perIoData->operationType == 0) {  // 接收操作
            perIoData->bytesTransferred = bytesTransferred;

            // 回显数据
            PER_IO_DATA* sendData = (PER_IO_DATA*)malloc(sizeof(PER_IO_DATA));
            ZeroMemory(sendData, sizeof(PER_IO_DATA));
            sendData->operationType = 1;
            sendData->socket = perIoData->socket;
            memcpy(sendData->buffer, perIoData->buffer, bytesTransferred);
            sendData->wsaBuf.buf = sendData->buffer;
            sendData->wsaBuf.len = bytesTransferred;

            WSASend(sendData->socket, &sendData->wsaBuf, 1, NULL, 0, (LPWSAOVERLAPPED)sendData, NULL);

            // 继续投递接收请求
            perIoData->wsaBuf.len = MAXBUFFERSIZE;
            perIoData->wsaBuf.buf = perIoData->buffer;
            DWORD flags = 0;
            WSARecv(perIoData->socket, &perIoData->wsaBuf, 1, NULL, &flags, (LPWSAOVERLAPPED)perIoData, NULL);
        } else {  // 发送操作完成
            free(perIoData);
        }
    }
    return 0;
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    WSADATA wsaData;
    SOCKET listenSocket, clientSocket;
    SOCKADDR_IN serverAddr, clientAddr;
    HANDLE hIOCP;
    SYSTEM_INFO sysInfo;
    DWORD i;

    // 初始化Winsock
    WSAStartup(MAKEWORD(2, 2), &wsaData);

    // 创建完成端口
    hIOCP = CreateIoCompletionPort(INVALID_HANDLE_VALUE, NULL, 0, 0);

    // 创建监听套接字
    listenSocket = WSASocket(AF_INET, SOCK_STREAM, 0, NULL, 0, WSA_FLAG_OVERLAPPED);
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    serverAddr.sin_port = htons(8080);
    bind(listenSocket, (SOCKADDR*)&serverAddr, sizeof(serverAddr));
    listen(listenSocket, SOMAXCONN);

    // 关联监听套接字到完成端口
    CreateIoCompletionPort((HANDLE)listenSocket, hIOCP, 0, 0);

    // 创建工作线程（通常根据CPU核心数）
    GetSystemInfo(&sysInfo);
    for (i = 0; i < sysInfo.dwNumberOfProcessors * 2; ++i) {
        CreateThread(NULL, 0, Ahri::WorkerThread, hIOCP, 0, NULL);
    }

    printf("Server running on port 8080...\n");

    while (TRUE) {
        int clientAddrLen = sizeof(clientAddr);
        clientSocket = accept(listenSocket, (SOCKADDR*)&clientAddr, &clientAddrLen);

        // 关联客户端套接字到完成端口
        CreateIoCompletionPort((HANDLE)clientSocket, hIOCP, 0, 0);

        // 投递初始接收请求
        Ahri::PER_IO_DATA* perIoData = (Ahri::PER_IO_DATA*)malloc(sizeof(Ahri::PER_IO_DATA));
        ZeroMemory(perIoData, sizeof(Ahri::PER_IO_DATA));
        perIoData->socket = clientSocket;
        perIoData->operationType = 0;
        perIoData->wsaBuf.buf = perIoData->buffer;
        perIoData->wsaBuf.len = Ahri::MAXBUFFERSIZE;

        DWORD flags = 0;
        WSARecv(clientSocket, &perIoData->wsaBuf, 1, NULL, &flags, (LPWSAOVERLAPPED)perIoData, NULL);
    }

    closesocket(listenSocket);
    WSACleanup();
    return 0;
}
