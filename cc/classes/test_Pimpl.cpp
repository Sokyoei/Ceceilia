#include <gtest/gtest.h>

#include "Pimpl.hpp"

int test_Pimpl() {
    Ahri::Dog p;
    p.print_info();
    return 0;
}

TEST(test_Pimpl, gtest_Pimpl) {
    EXPECT_EQ(0, test_Pimpl());
}
