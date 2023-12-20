/**
 * @file simd.c
 * @date 2023/12/13
 * @author Sokyoei
 * @details
 * SIMD(Single Instruction Multiple Data)「单指令集多数据」
 */

#include <stdio.h>

#include <intrin.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// load/store/set

void test_256_double() {
    double d[4];
    __m256d a = _mm256_set_pd(1.1, 2.2, 3.3, 4.4);
    __m256d b = _mm256_set_pd(1.1, 2.2, 3.3, 4.4);
    __m256d c = _mm256_add_pd(a, b);
    _mm256_store_pd(d, c);
    for (int i = 0; i < 4; i++) {
        printf("%f ", d[i]);
    }
    printf("\n");
}

#ifdef __cplusplus
}
#endif  // __cplusplus

int main(int argc, char const* argv[]) {
    test_256_double();
    return 0;
}
