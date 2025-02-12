#include <folly/String.h>

int main(int argc, char const* argv[]) {
    folly::fbstring str{"Ahri"};
    fmt::println(str);
    return 0;
}
