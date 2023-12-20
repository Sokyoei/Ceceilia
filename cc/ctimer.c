/**
 * @file ctimer.c
 * @date 2023/12/20
 * @author Sokyoei
 * @details
 * C time
 */

#include <stdio.h>
#include <time.h>

void get_current_time(char* str_time, char* format) {
    time_t now = time(NULL);
    printf("%d\n", now);
    struct tm* tm_now = localtime(&now);
    strftime(str_time, 20, format, tm_now);
    printf("%s\n ", str_time);
}

int main(int argc, char const* argv[]) {
    char now[20];
    get_current_time(now, "%Y-%m-%d %H:%M:%S");
    return 0;
}
