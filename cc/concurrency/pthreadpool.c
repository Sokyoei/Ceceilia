#include "pthreadpool.h"

#define NUMBER 2

void* manager(void* args);

void* worker(void* args);

void thread_exit(PThreadPool* pool);

PThreadPool* pthreadpool_create(int min_num, int max_num, int queue_size) {
    PThreadPool* pool = (PThreadPool*)malloc(sizeof(PThreadPool));

    do {
        if (pool == NULL) {
            printf("create PThreadPool fail\n");
            break;
        }
        pool->workers = (pthread_t*)malloc(sizeof(pthread_t) * max_num);
        if (pool->workers == NULL) {
            printf("create workers fail\n");
            break;
        }
        memset(pool->workers, 0, sizeof(pthread_t) * max_num);
        pool->min_num = min_num;
        pool->max_num = max_num;
        pool->busy_num = 0;
        pool->live_num = min_num;
        pool->exit_num = 0;

        if (pthread_mutex_init(&pool->lock_pool, NULL) != 0 || pthread_mutex_init(&pool->lock_busy, NULL) != 0 ||
            pthread_cond_init(&pool->cond_not_full, NULL) != 0 || pthread_cond_init(&pool->cond_not_empty, NULL) != 0) {
            printf("create lock or cond fail\n");
            break;
        }
        pool->tasks = (Task*)malloc(sizeof(Task) * queue_size);
        pool->queue_capacity = queue_size;
        pool->queue_size = 0;
        pool->queue_front = 0;
        pool->queue_rear = 0;

        pool->stop = false;

        // create thread
        pthread_create(&pool->manager, NULL, manager, pool);
        for (int i = 0; i < min_num; i++) {
            pthread_create(&pool->workers[i], NULL, worker, pool);
        }
        return pool;
    } while (0);

    // free
    if (pool && pool->workers) {
        free(pool->workers);
    }
    if (pool && pool->tasks) {
        free(pool->tasks);
    }
    if (pool) {
        free(pool);
    }
    return NULL;
}

void* worker(void* args) {
    PThreadPool* pool = (PThreadPool*)args;
    while (true) {
        pthread_mutex_lock(&pool->lock_pool);

        while (pool->queue_size == 0 && !pool->stop) {
            pthread_cond_wait(&pool->cond_not_empty, &pool->lock_pool);

            if (pool->exit_num > 0) {
                pool->exit_num--;
                if (pool->live_num > pool->max_num) {
                    pool->live_num--;
                    pthread_mutex_unlock(&pool->lock_pool);
                    // pthread_exit(NULL);
                    thread_exit(pool);
                }
            }
        }
        if (pool->stop) {
            pthread_mutex_unlock(&pool->lock_pool);
            // pthread_exit(NULL);
            thread_exit(pool);
        }
        Task task;
        task.function = pool->tasks[pool->queue_front].function;
        task.args = pool->tasks[pool->queue_front].args;
        pool->queue_front = (pool->queue_front + 1) & pool->queue_size;  // move queue front
        pool->queue_size--;

        pthread_cond_signal(&pool->cond_not_full);
        pthread_mutex_unlock(&pool->lock_pool);

        printf("thread %ld start working\n", pthread_self());
        pthread_mutex_lock(&pool->lock_busy);
        pool->busy_num++;
        pthread_mutex_unlock(&pool->lock_busy);
        task.function(task.args);
        free(task.args);
        task.args = NULL;

        printf("thread %ld end working\n", pthread_self());
        pthread_mutex_lock(&pool->lock_busy);
        pool->busy_num--;
        pthread_mutex_unlock(&pool->lock_busy);
    }
    return NULL;
}

void* manager(void* args) {
    PThreadPool* pool = (PThreadPool*)args;
    while (!pool->stop) {
        _sleep(3000);  // check by 3s

        pthread_mutex_lock(&pool->lock_pool);
        int queue_size = pool->queue_size;
        int live_num = pool->live_num;
        pthread_mutex_unlock(&pool->lock_pool);

        pthread_mutex_lock(&pool->lock_busy);
        int busy_num = pool->busy_num;
        pthread_mutex_unlock(&pool->lock_busy);

        // add thread
        if (queue_size > live_num && live_num < pool->max_num) {
            int counter = 0;
            pthread_mutex_lock(&pool->lock_pool);
            for (int i = 0; i < pool->max_num && counter < NUMBER && pool->live_num < pool->max_num; i++) {
                if (pool->workers[i] == 0) {
                    pthread_create(&pool->workers[i], NULL, worker, pool);
                    counter++;
                    pool->live_num++;
                }
            }
            pthread_mutex_unlock(&pool->lock_pool);
        }

        // destroy thread
        if (busy_num * 2 < pool->live_num && live_num > pool->min_num) {
            pthread_mutex_lock(&pool->lock_pool);
            pool->exit_num = NUMBER;
            pthread_mutex_unlock(&pool->lock_pool);

            for (int i = 0; i < NUMBER; i++) {
                pthread_cond_signal(&pool->cond_not_empty);
            }
        }
    }
    return NULL;
}

void thread_exit(PThreadPool* pool) {
    pthread_t tid = pthread_self();
    for (int i = 0; i < pool->max_num; i++) {
        if (pool->workers[i] == tid) {
            pool->workers[i] = 0;
            break;
        }
    }
    pthread_exit(NULL);
}

void pthreadpool_add(PThreadPool* pool, void (*func)(void*), void* args) {
    pthread_mutex_lock(&pool->lock_pool);
    while (pool->queue_size == pool->queue_capacity && !pool->stop) {
        pthread_cond_wait(&pool->cond_not_full, &pool->lock_pool);
    }
    if (pool->stop) {
        pthread_mutex_unlock(&pool->lock_pool);
        return;
    }

    // add task
    pool->tasks[pool->queue_rear].function = func;
    pool->tasks[pool->queue_rear].args = args;
    pool->queue_rear = (pool->queue_rear + 1) % pool->queue_capacity;
    pool->queue_size++;

    pthread_cond_signal(&pool->cond_not_empty);

    pthread_mutex_unlock(&pool->lock_pool);
}

int get_pthreadpool_busy_num(PThreadPool* pool) {
    pthread_mutex_lock(&pool->lock_busy);
    int busy_num = pool->busy_num;
    pthread_mutex_unlock(&pool->lock_busy);
    return busy_num;
}

int get_pthreadpool_alive_num(PThreadPool* pool) {
    pthread_mutex_lock(&pool->lock_pool);
    int alive_num = pool->live_num;
    pthread_mutex_unlock(&pool->lock_pool);
    return alive_num;
}

int pthreadpool_destroy(PThreadPool* pool) {
    if (pool == NULL) {
        return -1;
    }
    pool->stop = true;
    pthread_join(pool->manager, NULL);
    for (int i = 0; i < pool->live_num; i++) {
        pthread_cond_signal(&pool->cond_not_empty);
    }
    if (pool->tasks) {
        free(pool->tasks);
    }
    if (pool->workers) {
        free(pool->workers);
    }
    pthread_mutex_destroy(&pool->lock_pool);
    pthread_mutex_destroy(&pool->lock_busy);
    pthread_cond_destroy(&pool->cond_not_empty);
    pthread_cond_destroy(&pool->cond_not_full);
    free(pool);
    pool = NULL;
    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void task_func(void* args) {
    int num = *(int*)args;
    printf("thread %ld is working, number = %d\n", pthread_self(), num);
    _sleep(1000);
}

int main(int argc, char const* argv[]) {
    PThreadPool* pool = pthreadpool_create(3, 10, 100);
    for (int i = 0; i < 100; i++) {
        int* num = (int*)malloc(sizeof(int));
        *num = i + 100;
        pthreadpool_add(pool, task_func, num);
    }
    _sleep(30000);
    pthreadpool_destroy(pool);
    // free(num);

    return 0;
}
