/**
 * @file cstrings.c
 * @date 2023/12/21
 * @author Sokyoei
 * @details
 * C string
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char const* argv[]) {
    int len;
    char str[100];
    char* hello = "hello";
    printf("hello len = %llu\n", strlen(hello));

    // sprintf()
    len = sprintf(str, "%d %d", 12, 12);
    printf("str len = %d\n", len);
    printf("str = %s\n", str);

    // snprintf()
    char test_snprintf[100];
    len = snprintf(test_snprintf, 12, "%d %d", 12, 12);
    printf("str len = %d\n", len);
    printf("str = %s\n", str);

    // sprintf_s()
    // swprintf()
    // swprintf_s()
    return 0;
}
