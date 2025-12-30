#include "simple.hpp"

void Ahri::SimpleController::asyncHandleHttpRequest(const drogon::HttpRequestPtr& req,
                                            std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    // write your application logic here
    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setBody("<p>Hello, world!</p>");
    resp->setExpiredTime(0);
    callback(resp);
}
