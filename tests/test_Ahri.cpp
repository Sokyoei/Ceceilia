#include <gtest/gtest.h>

int add(int a, int b) {
    return a + b;
}

TEST(add, test_add) {
    ASSERT_EQ(add(1, 2), 4);
}
