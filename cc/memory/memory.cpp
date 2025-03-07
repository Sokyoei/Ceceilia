#include <cmath>
#include <iostream>

namespace Ahri {
int global_value = 10;

double func(double a, double b) {
    double value = std::pow(a, 3) + a * 3 + b;
    return value;
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    return 0;
}
