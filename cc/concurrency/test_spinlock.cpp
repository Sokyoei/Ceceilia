#include <iostream>
#include <thread>

#include "spinlock.hpp"

void test_spinlock() {
    Ahri::SpinLock spinlock;
    std::thread t1([&spinlock]() {
        spinlock.lock();
        for (int i = 0; i < 3; i++) {
            std::cout << "*";
        }
        std::cout << "\n";
        spinlock.unlock();
    });

    std::thread t2([&spinlock]() {
        spinlock.lock();
        for (int i = 0; i < 3; i++) {
            std::cout << "?";
        }
        std::cout << "\n";
        spinlock.unlock();
    });

    t1.join();
    t2.join();
}

int main(int argc, char const* argv[]) {
    test_spinlock();
    return 0;
}
