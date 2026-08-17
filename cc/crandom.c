#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char const* argv[]) {
    int r1;
    float r2;
    srand((int)time(NULL));
    // r = rand() % 31 + 20; // 生成 20 ~ 50 之间随机数
    r1 = rand() % 21 * 5;              // 生成 0 ~ 100 之间 5 的倍数
    r2 = (rand() % 901 + 100) * 0.01;  // 生成 1 ~ 10 之间小数，精确到小数点后 2 位
    printf("r1 = %d\n", r1);
    printf("r2 = %f\n", r2);

    return 0;
}
