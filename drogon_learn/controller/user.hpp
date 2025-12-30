#pragma once
#ifndef DROGON_LEARN_CONTROLLER_USER_HPP
#define DROGON_LEARN_CONTROLLER_USER_HPP

#include <drogon/HttpController.h>
#include <drogon/HttpTypes.h>

namespace Ahri {
class UserController : public drogon::HttpController<UserController> {
public:
    METHOD_LIST_BEGIN
    METHOD_ADD(UserController::get, "/", drogon::HttpMethod::Get);
    METHOD_ADD(UserController::post, "/", drogon::HttpMethod::Post);
    METHOD_ADD(UserController::put, "/", drogon::HttpMethod::Put);
    METHOD_ADD(UserController::del, "/", drogon::HttpMethod::Delete);
    METHOD_LIST_END

    void get(const drogon::HttpRequestPtr& req, std::function<void(const drogon::HttpResponsePtr&)>&& callback);
    void post(const drogon::HttpRequestPtr& req, std::function<void(const drogon::HttpResponsePtr&)>&& callback);
    void put(const drogon::HttpRequestPtr& req, std::function<void(const drogon::HttpResponsePtr&)>&& callback);
    void del(const drogon::HttpRequestPtr& req, std::function<void(const drogon::HttpResponsePtr&)>&& callback);
};
}  // namespace Ahri

#endif  // !DROGON_LEARN_CONTROLLER_USER_HPP
