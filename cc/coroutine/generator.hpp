/**
 * @file generator.hpp
 * @date 2024/03/22
 * @author Sokyoei
 *
 *
 */

#include <coroutine>

namespace Ahri {
template <std::movable T>
class Generator {
public:
    struct promise_type {
        Generator<T> get_return_object() { return Generator{_handle.from_promise(*this)}; }
        static std::suspend_always initial_suspend() noexcept { return {}; }
        static std::suspend_always final_suspend() noexcept { return {}; }
        std::suspend_always yield_value(T value) noexcept {}
        static void unhandled_exception() { throw; }
    };

    using handle_type = std::coroutine_handle<promise_type>;

    explicit Generator(handle_type handle) : _handle(handle) {}
    ~Generator() {}

private:
    handle_type _handle;
};
}  // namespace Ahri
