/**
 * @file pthreadpool.h
 * @date 2023/12/12
 * @author Sokyoei
 * @details
 * C pthread based ThreadPool
 */

#pragma once
#ifndef PTHREADPOOL_H
#define PTHREADPOOL_H

#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Ahri/Utils.h"

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

    pthread_t manager;              // 管理者线程
    pthread_t* workers;             // 工作线程
    pthread_mutex_t lock_pool;      // 线程池锁
    pthread_mutex_t lock_busy;      // 正在工作的线程锁
    pthread_cond_t cond_not_full;   // 任务队列是否满了
    pthread_cond_t cond_not_empty;  // 任务队列是否空了

    int min_num;   // 最大线程数
    int max_num;   // 最小线程数
    int busy_num;  // 正在工作线程数
    int live_num;  // 存活线程数
    int exit_num;  // 退出线程数

    bool stop;  // 线程池是否停止
} PThreadPool;

/**
 * @brief pthread 线程池创建
 * @param min_num 最小线程数
 * @param max_num 最大线程数
 * @param queue_size 任务队列大小
 * @return PThreadPool* 线程池
 */
PThreadPool* pthreadpool_create(int min_num, int max_num, int queue_size);

/**
 * @brief 线程池添加任务
 * @param pool 线程池
 * @param func 任务函数
 * @param args 任务函数的参数
 */
void pthreadpool_add(PThreadPool* pool, void (*func)(void*), void* args);

/**
 * @brief 获取线程池正在工作的线程数
 * @param pool 线程池
 * @return int 正在工作的线程数
 */
int get_pthreadpool_busy_num(PThreadPool* pool);

/**
 * @brief 获取线程池存活的线程数
 * @param pool 线程池
 * @return int 存活的线程数
 */
int get_pthreadpool_alive_num(PThreadPool* pool);

/**
 * @brief 销毁线程池
 * @param pool 线程池
 * @return int 状态信息
 */
int pthreadpool_destroy(PThreadPool* pool);

#ifdef __cplusplus
}
#endif

#endif  // PTHREADPOOL_H
