#pragma once
#ifndef AHRI_CECEILIA_UTILS_FMT_HPP
#define AHRI_CECEILIA_UTILS_FMT_HPP

#include <fmt/core.h>

namespace Ahri {
#define AHRI_FMT_FORMATTER_OSTREAM(Object)                         \
    FMT_BEGIN_NAMESPACE                                            \
    template <>                                                    \
    struct formatter<Object> {                                     \
        constexpr auto parse(format_parse_context& ctx) {          \
            return ctx.begin();                                    \
        }                                                          \
                                                                   \
        template <typename FormatContext>                          \
        auto format(const Object& obj, FormatContext& ctx) const { \
            std::ostringstream oss;                                \
            oss << obj;                                            \
            return fmt::format_to(ctx.out(), "{}", oss.str());     \
        }                                                          \
    };                                                             \
    FMT_END_NAMESPACE
}  // namespace Ahri

#endif  // !AHRI_CECEILIA_UTILS_FMT_HPP
