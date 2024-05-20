#include <gtest/gtest.h>

int add(int a, int b) {
    return a + b;
}

TEST(add, test_add) {
    EXPECT_EQ(add(2, 2), 4);
}
