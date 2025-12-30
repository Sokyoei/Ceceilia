#pragma once
#ifndef DROGON_LEARN_CONTROLLER_WS_HPP
#define DROGON_LEARN_CONTROLLER_WS_HPP

#include <drogon/WebSocketController.h>

namespace Ahri {
class Websocket : public drogon::WebSocketController<Websocket> {
public:
    WS_PATH_LIST_BEGIN
    WS_PATH_ADD("/ws", drogon::HttpMethod::Get);
    WS_PATH_LIST_END

    void handleNewMessage(const drogon::WebSocketConnectionPtr& wsconn_ptr,
                          std::string&& message,
                          const drogon::WebSocketMessageType& type);
    void handleNewConnection(const drogon::HttpRequestPtr& req, const drogon::WebSocketConnectionPtr& wsconn_ptr);
    void handleConnectionClosed(const drogon::WebSocketConnectionPtr& wsconn_ptr);
};
}  // namespace Ahri

#endif  // !DROGON_LEARN_CONTROLLER_WS_HPP
