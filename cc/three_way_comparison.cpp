/**
 * @file three_way_comparison.cpp
 * @date 2024/06/17
 * @author Sokyoei
 * @details
 * C++20 <=>
 */

#include <compare>
#include <iostream>

namespace Ahri {
class Comparer {
public:
    Comparer(int x, int y) : _x(x), _y(y) {}
    ~Comparer() {}
    // 默认会生成 ==, !=, >, <, >=, <= 6个运算符
    auto operator<=>(const Comparer&) const = default;

private:
    int _x;
    int _y;
};
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    Ahri::Comparer a{1, 2}, b{1, 4};
    std::cout << std::boolalpha << (a > b) << std::endl;
    return 0;
}
