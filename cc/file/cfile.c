/**
 * @file cfile.c
 * @date 2024/01/26
 * @author Sokyoei
 * @details
 * C file operation
 *
 */

// fopen()
// fopen64()
// fopen_s()

#include <stdio.h>

int main(int argc, char const* argv[]) {
    // write
    FILE* log_file = fopen("tempCodeRunnerFile.log", "w");
    if (log_file) {
        fprintf(log_file, "[%s %s][%s][line:%d]: log\n", __DATE__, __TIME__, __FILE__, __LINE__);
    }
    // fflush(log_file);
    fclose(log_file);

    // read
    return 0;
}
