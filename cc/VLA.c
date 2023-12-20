/**
 * @file VLA.c
 * @date 2023/12/20
 * @author Sokyoei
 * @details
 * C VLA(Variable Length Array)「变长数组(运行时决定)」
 */

// https://en.cppreference.com/w/c/language/array

#include <stdio.h>

int main(int argc, char const* argv[]) {
#ifdef __GNUC__
    int n = 10;
    int table[n];  // 先声明，再赋值
    for (int i = 0; i < n; i++) {
        table[i] = i;
    }

    for (int i = 0; i < n; i++) {
        printf("%d ", table[i]);
    }
    printf("\n");
#endif  // __GNUC__
    return 0;
}
