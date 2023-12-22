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
    char str[100];
    char* hello = "hello";
    printf("hello len = %d\n", strlen(hello));

    int len = sprintf(str, "%d %d", 12, 12);
    printf("str len = %d\n", len);
    printf("str = %s\n", str);
    return 0;
}
