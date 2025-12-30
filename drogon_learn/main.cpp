#include <drogon/drogon.h>

#include "controller/simple.hpp"
#include "controller/user.hpp"
#include "controller/ws.hpp"

int main(int argc, char const* argv[]) {
    drogon::app().registerHandler(
        "/", [](const drogon::HttpRequestPtr& req, std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setBody("<h1>hello world<h1>");
            callback(resp);
        });

    // Register Controller
    Ahri::SimpleController::initPathRouting();
    Ahri::UserController::initPathRouting();
    Ahri::Websocket::initPathRouting();

    // Application start
    drogon::app()
        // .loadConfigFile("./drogon.config.yaml")
        .setLogPath("./")
        .setLogLevel(trantor::Logger::kDebug)
        .addListener("0.0.0.0", 8000)
        .setThreadNum(16)
#ifndef _WIN32
        .enableRunAsDaemon()
#endif
        .run();
}
