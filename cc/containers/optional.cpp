#include <iostream>
#include <optional>
#include <random>

namespace Ahri {
std::optional<bool> maybe_has_value() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> uniform_int(1, 6);

    if (uniform_int(gen) > 3) {
        return true;
    }
    return std::nullopt;
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    auto value = Ahri::maybe_has_value();
    std::cout << "value is ";
    if (value.has_value()) {
        std::cout << value.value() << '\n';
    } else {
        std::cout << "null\n";
    }

    return 0;
}
