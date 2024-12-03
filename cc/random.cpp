/**
 * @file random.cpp
 * @date 2024/12/03
 * @author Sokyoei
 *
 *
 */

#include <iostream>
#include <random>

int main(int argc, char const* argv[]) {
    std::random_device rd;
    std::mt19937 gen(rd());

    // 均匀分布
    std::uniform_int_distribution<> uniform_int(1, 6);
    std::uniform_real_distribution<> uniform_real(1, 6);

    // 正态分布
    std::normal_distribution<> normal(0, 1);

    // 伯努利分布
    std::bernoulli_distribution bernoulli;
    
    for (int i = 0; i < 10; i++) {
        std::cout << uniform_int(gen) << '\n';
    }

    return 0;
}
