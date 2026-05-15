#include "ws.hpp"

void Ahri::Websocket::handleNewMessage(const drogon::WebSocketConnectionPtr& wsconn_ptr,
                                       std::string&& message,
                                       const drogon::WebSocketMessageType& type) {
    wsconn_ptr->send(message);
}

void Ahri::Websocket::handleNewConnection(const drogon::HttpRequestPtr& req,
                                          const drogon::WebSocketConnectionPtr& wsconn_ptr) {}

void Ahri::Websocket::handleConnectionClosed(const drogon::WebSocketConnectionPtr& wsconn_ptr) {}
