#include "user.hpp"

void Ahri::UserController::get(const drogon::HttpRequestPtr& req,
                               std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setBody("UserController get");
    callback(resp);
}

void Ahri::UserController::post(const drogon::HttpRequestPtr& req,
                                std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setBody("UserController post");
    callback(resp);
}

void Ahri::UserController::put(const drogon::HttpRequestPtr& req,
                               std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setBody("UserController put");
    callback(resp);
}

void Ahri::UserController::del(const drogon::HttpRequestPtr& req,
                               std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    callback(drogon::HttpResponse::newHttpResponse());
    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setBody("UserController del");
    callback(resp);
}
