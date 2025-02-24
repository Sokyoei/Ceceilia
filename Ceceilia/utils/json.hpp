/**
 * @file json.hpp
 * @date 2025/02/22
 * @author Sokyoei
 *
 * @see https://www.bilibili.com/video/BV1pa4y1g7v6
 */

#pragma once
#ifndef JSON_HPP
#define JSON_HPP

#include <charconv>
#include <optional>
#include <ostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

#include "Ahri.hpp"

namespace Ahri {
struct JSONObject {
    std::variant<std::nullptr_t,
                 bool,
                 int,
                 double,
                 std::string,
                 std::vector<JSONObject>,
                 std::unordered_map<std::string, JSONObject>>
        inner;

    friend std::ostream& operator<<(std::ostream& os, const JSONObject& json) {}
};

template <typename T>
std::optional<T> try_parse_num(std::string_view str) {
    T value;
    std::from_chars_result res = std::from_chars(str.data(), str.data() + str.size(), value);
    if (res.ec == std::errc() && res.ptr == str.data() + str.size()) {
        return value;
    }
    return std::nullopt;
}

char unescaped_char(char c) {
    // clang-format off
    switch (c) {
        case 'n': return '\n';
        case 'r': return '\r';
        case '0': return '\0';
        case 't': return '\t';
        case 'v': return '\v';
        case 'f': return '\f';
        case 'b': return '\b';
        case 'a': return '\a';
        default: return c;
    }
    // clang-format on
}

JSONObject parse(std::string_view json) {
    if (json.empty()) {
        return JSONObject{std::nullptr_t{}};
    } else if ('0' <= json[0] && json[0] <= '9' || json[0] == '-' || json[0] == '+') {
        std::regex num_re{"[+-]?[0-9]+(\\.[0-9]*)?([eE][+-]?[0-9]+)?"};
        std::cmatch match;
        if (std::regex_search(json.data(), json.data() + json.size(), match, num_re)) {
            std::string str = match.str();
            // if (auto num = try_parse_num<int>(str); num.has_value()) {
            //     return JSONObject{num.value()};
            // }
            // if (auto num = try_parse_num<double>(str); num.has_value()) {
            //     return JSONObject{num.value()};
            // }

            // std::optional.has_value() == bool(std::optional)
            if (auto num = try_parse_num<int>(str)) {
                return JSONObject{*num};  // fast than num.value(), because not if judgment
            }
            if (auto num = try_parse_num<double>(str)) {
                return JSONObject{*num};
            }
        }
    } else if (json[0] == '"') {
        // size_t next_comma = json.find('"', 1);
        // std::string str{json.substr(1, next_comma - 1)};

        std::string str;
        enum { Raw, Escaped } phase = Raw;
        for (size_t i = 1; i < json.size(); i++) {
            char ch = json[i];
            if (phase == Raw) {
                if (ch == '\\') {
                    phase = Escaped;
                } else if (ch == '"') {
                    break;
                } else {
                    str += ch;
                }

            } else if (phase == Escaped) {
                str += unescaped_char(ch);
                phase = Raw;
            }
        }
        return JSONObject{std::move(str)};
    } else if (json[0] == '[') {
    }

    return JSONObject{std::nullptr_t{}};
}
}  // namespace Ahri

#endif  // !JSON_HPP
