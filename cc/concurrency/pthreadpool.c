#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct Task {
    void* (*function)(void*);
    void* args;
} Task;

typedef struct PThreadPool {
    pthread_t* workers;
    bool stop;
} PThreadPool;

int main(int argc, char const* argv[]) {
    return 0;
}
