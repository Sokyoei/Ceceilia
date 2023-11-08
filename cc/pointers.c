#include <stdio.h>
#include <stdlib.h>

// array pointer
typedef int (*int_x3_arr_t)[3];

// function pointer
typedef int (*int_any_fn_t)();

int func(int a) {
    return a;
}

int main(int argc, char const* argv[]) {
    int_any_fn_t int_any_fn_ptr = func;
    printf("%d", int_any_fn_ptr(1));

    return 0;
}
