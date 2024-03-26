/**
 * @file tuple.cpp
 * @date 2024/03/26
 * @author Sokyoei
 *
 *
 */

#include <iostream>
#include <tuple>
#include <utility>

int main(int argc, char* argv[]) {
    std::tuple<int, char> t(20, 'a');
    std::cout << std::get<0>(t) << std::get<1>(t) << std::tuple_size<decltype(t)>::value << std::endl;
    std::get<0>(t) = 100;

    auto t2 = std::make_tuple("ahri", 21, 74.5);
    const char* str = nullptr;
    int a;
    double b;
    std::tie(str, a, b) = t2;
    auto cat = std::tuple_cat(t, t2);
    std::cout << std::tuple_size<decltype(cat)>::value << std::endl;

    return 0;
}
