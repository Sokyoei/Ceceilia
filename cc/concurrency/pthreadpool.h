#pragma once
#ifndef PTHREADPOOL_H
#define PTHREADPOOL_H

#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Task {
    void* (*function)(void*);
    void* args;
} Task;

typedef struct PThreadPool {
    pthread_mutex_t lock;
    pthread_cond_t cond;
    pthread_t* workers;
    Task* tasks;
    bool stop;
} PThreadPool;

PThreadPool* pthreadpool_create(int workers, int queue_size, int flags);

#ifdef __cplusplus
}
#endif

#endif  // PTHREADPOOL_H
