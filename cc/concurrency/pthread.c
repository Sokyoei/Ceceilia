#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

pthread_rwlock_t* rwlock;   // 读写锁
pthread_spinlock_t splock;  // 自旋锁
pthread_mutex_t* mutex;     // 互斥锁
pthread_cond_t* cond;       // 条件变量
pthread_barrier_t barr;     // 屏障锁

typedef struct Args {
    int num;
    const char* name;
} Args;

void* hello(void* args) {
    Args* _args = (Args*)args;
    for (int i = 0; i < 50; i++) {
        pthread_mutex_lock(mutex);
        printf("%s : %d\n", _args->name, _args->num);
        _args->num += 2;
        pthread_cond_wait(cond, mutex);
        pthread_cond_signal(cond);
        pthread_mutex_unlock(mutex);
    }
    return NULL;
}

void* hello2(void* args) {
    Args* _args = (Args*)args;
    for (int i = 0; i < 50; i++) {
        pthread_mutex_lock(mutex);
        printf("%s : %d\n", _args->name, _args->num);
        _args->num += 2;
        pthread_cond_signal(cond);
        pthread_cond_wait(cond, mutex);
        pthread_mutex_unlock(mutex);
    }
    return NULL;
}

int main(int argc, char const* argv[]) {
    mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
    cond = (pthread_cond_t*)malloc(sizeof(pthread_cond_t));

    pthread_mutex_init(mutex, NULL);
    pthread_cond_init(cond, NULL);

    pthread_t* t = (pthread_t*)malloc(sizeof(pthread_t));
    pthread_t* t2 = (pthread_t*)malloc(sizeof(pthread_t));
    Args a = {0, "Ahri"};
    Args s = {1, "Sokyoei"};
    int status = pthread_create(t, NULL, hello, (void*)&a);
    int status2 = pthread_create(t2, NULL, hello2, (void*)&s);
    pthread_join(*t, NULL);
    pthread_join(*t2, NULL);
    free(t);
    free(t2);
    pthread_mutex_destroy(mutex);
    free(mutex);
    free(cond);

    return 0;
}
