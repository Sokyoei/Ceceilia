#pragma once
#ifndef DRAGON_LEARN_UTILS_META_HPP
#define DRAGON_LEARN_UTILS_META_HPP

#include <drogon/HttpSimpleController.h>
#include <drogon/drogon.h>

namespace Ahri {
using Callable = std::function<void(const drogon::HttpResponsePtr&)>;
}

#endif  // !DRAGON_LEARN_UTILS_META_HPP
