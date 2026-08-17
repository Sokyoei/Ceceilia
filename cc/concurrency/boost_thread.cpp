#include <fmt/core.h>
#include <boost/thread.hpp>

namespace Ahri {
void print() {
    fmt::println("Hello from Boost Thread!");
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    boost::thread t(Ahri::print);
    t.join();  // Wait for the thread to finish
    return 0;
}
