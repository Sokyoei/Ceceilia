/**
 * @file singleton.cpp
 * @date 2024/03/22
 * @author Sokyoei
 *
 *
 */

#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

namespace Ahri {
class SingletonLazy;

class SingletonOnce {
private:
    SingletonOnce() = default;
    SingletonOnce(const SingletonOnce&) = delete;
    SingletonOnce& operator=(const SingletonOnce&) = delete;
    static std::shared_ptr<SingletonOnce> instance;

public:
    static std::shared_ptr<SingletonOnce> get_instance() {
        static std::once_flag flag;
        std::call_once(flag, [&]() { instance = std::shared_ptr<SingletonOnce>(new SingletonOnce); });
        return instance;
    }
};

class SingletonDeletor {
public:
    void operator()(SingletonLazy* sl) {
        std::cout << "SingletonDeletor delete sl" << std::endl;
        delete sl;
    }
};

/**
 * @brief 懒汉式单例
 */
class SingletonLazy {
private:
    SingletonLazy() {}
    ~SingletonLazy() { std::cout << "~SingletonLazy" << std::endl; }
    SingletonLazy(const SingletonLazy&) = delete;
    SingletonLazy& operator=(const SingletonLazy&) = delete;
    friend class SingletonDeletor;

public:
    static std::shared_ptr<SingletonLazy> get_instance() {
        if (singletonLazy != nullptr) {
            return singletonLazy;
        }
        // 加锁，防止多线程重复创建对象
        mutex.lock();
        if (singletonLazy != nullptr) {
            mutex.unlock();
            return singletonLazy;
        }
        // singletonLazy = std::make_shared<SingletonLazy>();
        // new 操作不是原子操作
        singletonLazy = std::shared_ptr<SingletonLazy>(new SingletonLazy, SingletonDeletor());
        mutex.unlock();
        return singletonLazy;
    }

private:
    static std::shared_ptr<SingletonLazy> singletonLazy;
    static std::mutex mutex;
};

/**
 * @brief 饿汉式单例，在主线程启动时创建单例对象，是从使用角度来规避多线程的安全问题
 */
class [[deprecated]] SingletonHungry {
private:
    SingletonHungry() {}
    SingletonHungry(const SingletonHungry&) = delete;
    SingletonHungry& operator=(const SingletonHungry&) = delete;

public:
    static SingletonHungry* get_instance() {
        if (singletonHungry == nullptr) {
            singletonHungry = new SingletonHungry();
        }
        return singletonHungry;
    }

private:
    static SingletonHungry* singletonHungry;
};

#if __cplusplus >= 201103L || _MSVC_LANG >= 201103L
/**
 * @brief 单例
 */
class Singleton {
private:
    Singleton() {}
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;

public:
    static Singleton& get_instance() {
        static Singleton singleton;
        std::cout << "Singleton" << std::endl;
        return singleton;
    }
};
#endif  // __cplusplus >= 201103L || _MSVC_LANG >= 201103L
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    auto& singletion = Ahri::Singleton::get_instance();
    return 0;
}
