#include <coroutine>

namespace Ahri {
template <std::movable T>
class Generator {
public:
    struct promise_type {
        Generator<T> get_return_object() { return Generator{handle.from_promise(*this)}; }
        static std::suspend_always initial_suspend() noexcept { return {}; }
        static std::suspend_always final_suspend() noexcept { return {}; }
        std::suspend_always yield_value(T value) noexcept {}
        static void unhandled_exception() { throw; }
    };

    explicit Generator(std::coroutine_handle<promise_type> handle) : handle(handle) {}
    ~Generator() {}

private:
    std::coroutine_handle<promise_type> handle;
};
}  // namespace Ahri
