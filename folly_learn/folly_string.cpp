#include <iostream>

#include <fmt/format.h>
#include <folly/String.h>

int main(int argc, const char** argv) {
    folly::fbstring str{"Ahri"};
#ifdef _MSC_VER
    fmt::println(str);
#elif defined(__GNUC__) || defined(__clang__)
    fmt::println("{}", static_cast<std::string>(str));
#else
#endif
    return 0;
}
