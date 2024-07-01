/**
 * @file ranges.cpp
 * @date 2024/06/17
 * @author Sokyoei
 * @details
 * C++20 ranges
 */

#include <algorithm>
#include <iostream>
#include <ranges>
#include <vector>

int main(int argc, char const* argv[]) {
    std::vector<int> vec{1, 2, 3, 4, 5};
    auto vec_filter = std::views::filter([](int n) { return n % 2 == 0; });
    auto vec_transform = std::views::transform([](int n) { return n * 2; });
    auto res = vec | vec_filter | vec_transform;

    for (auto&& i : res) {
        std::cout << i << std::endl;
    }

    auto range = std::views::iota(1) | std::views::drop(2) | std::views::take(3);
    using legacy_iterator = std::common_iterator<decltype(range.begin()), decltype(range.end())>;

    // std::copy(range.begin(), range.end(), std::ostream_iterator<int>(std::cout));
    std::copy(legacy_iterator(range.begin()), legacy_iterator(range.end()), std::ostream_iterator<int>(std::cout));
    std::ranges::copy(range, std::ostream_iterator<int>(std::cout));

    return 0;
}
