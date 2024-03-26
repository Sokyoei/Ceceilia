/**
 * @file smater_pointer.cpp
 * @date 2023/12/13
 * @author Sokyoei
 * @details
 * C++11 smart pointer
 */

#include <iostream>
#include <memory>

namespace Ahri {
void test_unique_ptr() {
    // std::unique_ptr<int[]> uptr(new int[20]); // method 1
    std::unique_ptr<int[]> uptr = std::make_unique<int[]>(20);  // method 2
    std::unique_ptr<int[]> uptr2 = std::move(uptr);             // std::unique_ptr is move-only
    std::cout << uptr.get() << std::endl;                       // then uptr is nullptr
    for (auto i = 0; i < 20; i++) {
        uptr2[i] = i * i;
    }
    for (auto i = 0; i < 20; i++) {
        std::cout << uptr2[i] << " ";
    }
    std::cout << std::endl;
}

void test_shared_ptr() {
    // std::shared_ptr<int> sptr(new int(1));
    std::shared_ptr<int> sptr = std::make_shared<int>(1);
    std::shared_ptr<int> sptrr(sptr);
    std::cout << sptr.use_count() << std::endl;
    sptrr.reset();
    std::cout << sptr.use_count() << std::endl;
    std::cout << sptrr.use_count() << std::endl;

    // std::shared_ptr<int> sptr2(new int[10], [](int* p) {delete[]p; });
    std::shared_ptr<int> sptr2(new int[10], std::default_delete<int[]>());
}

void test_weak_ptr() {}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    Ahri::test_unique_ptr();
    Ahri::test_shared_ptr();
    Ahri::test_weak_ptr();
    return 0;
}
