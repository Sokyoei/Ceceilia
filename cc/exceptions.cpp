/**
 * @file exceptions.cpp
 * @date 2023/12/13
 * @author Sokyoei
 * @details
 * C++ exception warning
 */

#include <cassert>

#include <any>
#include <chrono>
#include <exception>
#include <filesystem>
#include <format>
#include <future>
#include <ios>
#include <iostream>
#include <optional>
#include <regex>
#include <stdexcept>
#include <system_error>
#include <variant>

#include "config.h"

namespace Ahri {
/**
 * @brief Sokyoei'Error example
 */
class SokyoeiError : public std::exception {
private:
    const char* _message;

public:
    explicit SokyoeiError(const char* message) : _message(message) {}
    [[nodiscard]] const char* what() const noexcept override { return _message; }
};
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    try {
        throw Ahri::SokyoeiError("Something are Error");
    } catch (const Ahri::SokyoeiError& e) {
        std::cerr << e.what() << '\n';
    }

    return 0;
}

// #define ECXCEPTION
#ifdef ECXCEPTION

/**
 * @brief C++ exception
 */
#if __cpp_exceptions
class exception : public std::exception {
    class bad_alloc : public std::bad_alloc {  // new/new[] 分配内存失败
        class bad_array_new_length : public std::bad_array_new_length {};
    };
    class bad_cast : public std::bad_cast {                // dynamic_cast 转换失败
        class bad_any_cast : public std::bad_any_cast {};  // <any>
    };
    class bad_typeid : public std::bad_typeid {};  // typeid 抛出
    class bad_exception : public std::bad_exception {};
    class bad_function_call : public std::bad_function_call {};
    class bad_optional_access : public std::bad_optional_access {};
    class bad_weak_ptr : public std::bad_weak_ptr {};
    class bad_variant_access : public std::bad_variant_access {};
    // 逻辑错误
    class logic_error : public std::logic_error {
        class domain_error : public std::domain_error {};          // 参数值域错误
        class invalid_argument : public std::invalid_argument {};  // 参数不合适
        class length_error : public std::length_error {};          // 试图生成超过该类型的最大长度
        class out_of_range : public std::out_of_range {};          // 超出有效范围
        class future_error : public std::future_error {};
    };
    // 运行时错误
    class runtime_error : public std::runtime_error {
        class range_error : public std::range_error {};          // 计算结果超出有意义的值域范围
        class overflow_error : public std::overflow_error {};    // 算数计算上溢
        class underflow_error : public std::underflow_error {};  // 算数计算下溢
        class regex_error : public std::regex_error {};
        class system_error : public std::system_error {
            class failure : public std::ios_base::failure {};
            class filesystem_error : public std::filesystem::filesystem_error {};
        };
        class nonexistent_local_time : public std::chrono::nonexistent_local_time {};  // C++20
        class ambiguous_local_time : public std::chrono::ambiguous_local_time {};      // C++20
        class format_error : public std::format_error {};
    };
};
#endif  // __cpp_exceptions

/**
 * @brief POSIX errc (/usr/include/asm-generic/errno.h)
 */
void errc() {
    static_assert(static_cast<int>(std::errc::operation_not_permitted) == EPERM);          //
    static_assert(static_cast<int>(std::errc::no_such_file_or_directory) == ENOENT);       //
    static_assert(static_cast<int>(std::errc::no_such_process) == ESRCH);                  //
    static_assert(static_cast<int>(std::errc::interrupted) == EINTR);                      //
    static_assert(static_cast<int>(std::errc::io_error) == EIO);                           //
    static_assert(static_cast<int>(std::errc::no_such_device_or_address) == ENXIO);        //
    static_assert(static_cast<int>(std::errc::argument_list_too_long) == E2BIG);           // 参数列表过长
    static_assert(static_cast<int>(std::errc::executable_format_error) == ENOEXEC);        //
    static_assert(static_cast<int>(std::errc::bad_file_descriptor) == EBADF);              //
    static_assert(static_cast<int>(std::errc::no_child_process) == ECHILD);                //
    static_assert(static_cast<int>(std::errc::resource_unavailable_try_again) == EAGAIN);  // 资源不可用，须重试
    static_assert(static_cast<int>(std::errc::not_enough_memory) == ENOMEM);               //
    static_assert(static_cast<int>(std::errc::permission_denied) == EACCES);               // 权限被禁止
    static_assert(static_cast<int>(std::errc::bad_address) == EFAULT);                     //
    static_assert(static_cast<int>(std::errc::device_or_resource_busy) == EBUSY);          //
    static_assert(static_cast<int>(std::errc::file_exists) == EEXIST);                     //
    static_assert(static_cast<int>(std::errc::cross_device_link) == EXDEV);                //
    static_assert(static_cast<int>(std::errc::no_such_device) == ENODEV);                  //
    static_assert(static_cast<int>(std::errc::not_a_directory) == ENOTDIR);                //
    static_assert(static_cast<int>(std::errc::is_a_directory) == EISDIR);                  //
    static_assert(static_cast<int>(std::errc::invalid_argument) == EINVAL);                //
    static_assert(static_cast<int>(std::errc::too_many_files_open_in_system) == ENFILE);   //
    static_assert(static_cast<int>(std::errc::too_many_files_open) == EMFILE);             //
    static_assert(static_cast<int>(std::errc::inappropriate_io_control_operation) == ENOTTY);  //
    static_assert(static_cast<int>(std::errc::file_too_large) == EFBIG);                       //
    static_assert(static_cast<int>(std::errc::no_space_on_device) == ENOSPC);                  //
    static_assert(static_cast<int>(std::errc::invalid_seek) == ESPIPE);                        //
    static_assert(static_cast<int>(std::errc::read_only_file_system) == EROFS);                //
    static_assert(static_cast<int>(std::errc::too_many_links) == EMLINK);                      //
    static_assert(static_cast<int>(std::errc::broken_pipe) == EPIPE);                          //
    static_assert(static_cast<int>(std::errc::argument_out_of_domain) == EDOM);                //
    static_assert(static_cast<int>(std::errc::result_out_of_range) == ERANGE);                 //
    static_assert(static_cast<int>(std::errc::resource_deadlock_would_occur) == EDEADLK);      //
    static_assert(static_cast<int>(std::errc::filename_too_long) == ENAMETOOLONG);             //
    static_assert(static_cast<int>(std::errc::no_lock_available) == ENOLCK);                   //
    static_assert(static_cast<int>(std::errc::function_not_supported) == ENOSYS);              //
    static_assert(static_cast<int>(std::errc::directory_not_empty) == ENOTEMPTY);              //
    static_assert(static_cast<int>(std::errc::illegal_byte_sequence) == EILSEQ);               //
    static_assert(static_cast<int>(std::errc::address_in_use) == EADDRINUSE);                  // 地址在使用中
    static_assert(static_cast<int>(std::errc::address_not_available) == EADDRNOTAVAIL);        // 地址不可用
    static_assert(static_cast<int>(std::errc::address_family_not_supported) == EAFNOSUPPORT);  // 不支持地址系列
    static_assert(static_cast<int>(std::errc::connection_already_in_progress) == EALREADY);    //
    static_assert(static_cast<int>(std::errc::bad_message) == EBADMSG);                        //
    static_assert(static_cast<int>(std::errc::operation_canceled) == ECANCELED);               //
    static_assert(static_cast<int>(std::errc::connection_aborted) == ECONNABORTED);            //
    static_assert(static_cast<int>(std::errc::connection_refused) == ECONNREFUSED);            //
    static_assert(static_cast<int>(std::errc::connection_reset) == ECONNRESET);                //
    static_assert(static_cast<int>(std::errc::destination_address_required) == EDESTADDRREQ);  //
    static_assert(static_cast<int>(std::errc::host_unreachable) == EHOSTUNREACH);              //
    static_assert(static_cast<int>(std::errc::identifier_removed) == EIDRM);                   //
    static_assert(static_cast<int>(std::errc::operation_in_progress) == EINPROGRESS);          //
    static_assert(static_cast<int>(std::errc::already_connected) == EISCONN);                  //
    static_assert(static_cast<int>(std::errc::too_many_symbolic_link_levels) == ELOOP);        //
    static_assert(static_cast<int>(std::errc::message_size) == EMSGSIZE);                      //
    static_assert(static_cast<int>(std::errc::network_down) == ENETDOWN);                      //
    static_assert(static_cast<int>(std::errc::network_reset) == ENETRESET);                    //
    static_assert(static_cast<int>(std::errc::network_unreachable) == ENETUNREACH);            //
    static_assert(static_cast<int>(std::errc::no_buffer_space) == ENOBUFS);                    //
    static_assert(static_cast<int>(std::errc::no_message_available) == ENODATA);               //
    static_assert(static_cast<int>(std::errc::no_link) == ENOLINK);                            //
    static_assert(static_cast<int>(std::errc::no_message) == ENOMSG);                          //
    static_assert(static_cast<int>(std::errc::no_protocol_option) == ENOPROTOOPT);             //
    static_assert(static_cast<int>(std::errc::no_stream_resources) == ENOSR);                  //
    static_assert(static_cast<int>(std::errc::not_a_stream) == ENOSTR);                        //
    static_assert(static_cast<int>(std::errc::not_connected) == ENOTCONN);                     //
    static_assert(static_cast<int>(std::errc::state_not_recoverable) == ENOTRECOVERABLE);      //
    static_assert(static_cast<int>(std::errc::not_a_socket) == ENOTSOCK);                      //
    static_assert(static_cast<int>(std::errc::not_supported) == ENOTSUP);                      //
    static_assert(static_cast<int>(std::errc::operation_not_supported) == EOPNOTSUPP);         //
    static_assert(static_cast<int>(std::errc::value_too_large) == EOVERFLOW);                  //
    static_assert(static_cast<int>(std::errc::owner_dead) == EOWNERDEAD);                      //
    static_assert(static_cast<int>(std::errc::protocol_error) == EPROTO);                      //
    static_assert(static_cast<int>(std::errc::protocol_not_supported) == EPROTONOSUPPORT);     //
    static_assert(static_cast<int>(std::errc::wrong_protocol_type) == EPROTOTYPE);             //
    static_assert(static_cast<int>(std::errc::stream_timeout) == ETIME);                       //
    static_assert(static_cast<int>(std::errc::timed_out) == ETIMEDOUT);                        //
    static_assert(static_cast<int>(std::errc::text_file_busy) == ETXTBSY);                     //
    static_assert(static_cast<int>(std::errc::operation_would_block) == EWOULDBLOCK);          //
};

#pragma warning(disable : 4996)

#endif
