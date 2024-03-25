/**
 * @file pthreadpool_example.c
 * @date 2024/03/22
 * @author Sokyoei
 *
 *
 */

#include "pthreadpool.h"

void task_func(void* args) {
    int num = *(int*)args;
    printf("thread %ld is working, number = %d\n", pthread_self(), num);
    sleep(1);
}

int main(int argc, char const* argv[]) {
    PThreadPool* pool = pthreadpool_create(3, 10, 100);
    for (int i = 0; i < 100; i++) {
        int* num = (int*)malloc(sizeof(int));
        *num = i + 100;
        pthreadpool_add(pool, task_func, num);
    }
    sleep(30);
    pthreadpool_destroy(pool);

    return 0;
}
