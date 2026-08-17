#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <Windows.h>
#endif

/**
 * @brief 16进制字符串转 uint8_t 数组
 * @param hexstr 字符串
 * @param arr uint8_t 数组
 */
void hexstr_to_u8arr(char* hexstr, uint8_t* arr) {
    const int len = strlen(hexstr) / 2;
    // 为什么要用 int?
    int tmp;
    for (int i = 0; i < len; i++) {
        sscanf(hexstr, "%02x", &tmp);
        arr[i] = tmp;
        hexstr += 2;
    }
}

void u8arr2charr(uint8_t* arr, int arrlen, char* str) {
    memcpy(str, arr, arrlen);
    str[arrlen] = '\0';
}

void charr2u8arr(char* str, int strlen, uint8_t* arr) {
    memcpy(arr, str, strlen);
}

int main(int argc, char const* argv[]) {
#ifdef _WIN32
    system("chcp 65001");
#endif

    // 示例5: 16进制字符串 -> uint8_t -> char -> uint8_t -> char
    char original[] = "48656c6c6f20576f726c64";  // "Hello World" 的16进制
    uint8_t buffer1[20] = {0};
    char buffer2[20] = {0};
    uint8_t buffer3[20] = {0};
    char buffer4[20] = {0};

    // 16进制字符串 -> uint8_t
    hexstr_to_u8arr(original, buffer1);
    printf("16进制字符串: \"%s\"\n", original);
    printf("uint8_t: [");
    for (int i = 0; i < 12; i++) {
        printf("0x%02x", buffer1[i]);
        if (i < 11) {
            printf(", ");
        }
    }
    printf("]\n");

    // uint8_t -> 字符串
    u8arr2charr(buffer1, 12, buffer2);
    printf("uint8_t -> 字符串: \"%s\"\n", buffer2);

    // 字符串 -> uint8_t
    charr2u8arr(buffer2, strlen(buffer2), buffer3);
    printf("字符串 -> uint8_t: [");
    for (int i = 0; i < 12; i++) {
        printf("0x%02x", buffer3[i]);
        if (i < 11) {
            printf(", ");
        }
    }
    printf("]\n");

    // uint8_t -> 字符串
    u8arr2charr(buffer3, 12, buffer4);
    printf("uint8_t -> 字符串: \"%s\"\n", buffer4);

    return 0;
}
