#pragma once
#ifndef PTHREADPOOL_H
#define PTHREADPOOL_H

#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Task {
    void (*function)(void*);
    void* args;
} Task;

typedef struct PThreadPool {
    Task* tasks;         // 任务队列
    int queue_capacity;  // 队列容量
    int queue_size;      // 当前任务个数
    int queue_front;     // 队列头
    int queue_rear;      // 队列尾

    pthread_t manager;   // 管理者线程
    pthread_t* workers;  // 工作线程
    pthread_mutex_t lock_pool;
    pthread_mutex_t lock_busy;
    pthread_cond_t cond_not_full;
    pthread_cond_t cond_not_empty;

    int min_num;   // 最大线程数
    int max_num;   // 最小线程数
    int busy_num;  // 工作线程数
    int live_num;  // 存活线程数
    int exit_num;  // 退出线程数

    bool stop;
} PThreadPool;

PThreadPool* pthreadpool_create(int min_num, int max_num, int queue_size);

void pthreadpool_add(PThreadPool* pool, void (*func)(void*), void* args);

int get_pthreadpool_busy_num(PThreadPool* pool);

int get_pthreadpool_alive_num(PThreadPool* pool);

int pthreadpool_destroy(PThreadPool* pool);

#ifdef __cplusplus
}
#endif

#endif  // PTHREADPOOL_H
