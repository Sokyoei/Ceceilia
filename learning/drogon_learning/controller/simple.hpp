#pragma once
#ifndef DROGON_LEARN_CONTROLLER_SIMPLE_HPP
#define DROGON_LEARN_CONTROLLER_SIMPLE_HPP

#include <drogon/HttpSimpleController.h>

namespace Ahri {
class SimpleController : public drogon::HttpSimpleController<SimpleController> {
public:
    void asyncHandleHttpRequest(const drogon::HttpRequestPtr& req,
                                std::function<void(const drogon::HttpResponsePtr&)>&& callback) override;
    PATH_LIST_BEGIN
    PATH_ADD("/test", drogon::HttpMethod::Get);
    PATH_LIST_END
};
}  // namespace Ahri

#endif  // !DROGON_LEARN_CONTROLLER_SIMPLE_HPP
