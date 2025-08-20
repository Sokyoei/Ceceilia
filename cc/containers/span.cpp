/**
 * @file span.cpp
 * @date 2025/08/19
 * @author Sokyoei
 * @details std::span std::mdspan
 *
 */

#include "Ahri/Ahri.hpp"

#include <array>
#include <iostream>
#include <print>
#include <vector>
#include <version>

#if defined AHRI_CXX20 && defined __cpp_lib_span
#include <span>
#endif
#if defined AHRI_CXX23 && defined __cpp_lib_mdspan
#include <mdspan>
#endif

namespace Ahri {
#ifdef __cpp_lib_span
void old_print_span(std::span<int> s) {
    std::print("old style print span: ");
    for (size_t i = 0; i < s.size(); i++) {
        std::print("{} ", s[i]);
    }
    std::println("");
}

void range_for_span(std::span<int> s) {
    std::print("range for span: ");
    for (const auto& elem : s) {
        std::print("{} ", elem);
    }
    std::println("");
}

void span_example() {
    int arr[] = {1, 2, 3, 4, 5};
    std::span<int> span_from_array(arr);
    old_print_span(span_from_array);

    std::vector<int> vec = {10, 20, 30, 40, 50};
    std::span<int> span_from_vector(vec);
    range_for_span(span_from_vector);

    std::array<int, 5> std_arr = {100, 200, 300, 400, 500};
    std::span<int, 5> span_from_std_array(std_arr);
    std::println("`std::print` print `std::span`: {}", span_from_array);

    // subspan
    auto subspan = span_from_array.subspan(1, 3);
    std::println("arr subspan 1~3: {}", subspan);

    // first and last
    std::println("vec first 3: {}", span_from_vector.first(3));
    std::println("vec last 2: {}", span_from_vector.last(2));
}
#endif  // __cpp_lib_span
}  // namespace Ahri

int main() {
#ifdef __cpp_lib_span
    Ahri::span_example();
#endif  // __cpp_lib_span

    return 0;
}
