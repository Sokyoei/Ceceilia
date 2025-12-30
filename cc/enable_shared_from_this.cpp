#include <iostream>
#include <memory>

namespace Ahri {
class Connection : public std::enable_shared_from_this<Connection> {
public:
    void doSomething() {
        auto self = shared_from_this();
        std::cout << "shared_from_this() use_count: " << self.use_count() << std::endl;
    }
};
}  // namespace Ahri

int main() {
    auto conn = std::make_shared<Ahri::Connection>();
    conn->doSomething();
}
