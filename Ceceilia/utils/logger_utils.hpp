/**
 * @file logger_utils.hpp
 * @date 2024/08/22
 * @author Sokyoei
 * @details
 * logger
 */

#ifndef LOGGER_UTILS_HPP
#define LOGGER_UTILS_HPP

#include <initializer_list>

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace Ahri {
/**
 * @brief 初始化 logger
 * @param file_path 日志文件路径
 * @example
 *
 * ```cpp
 * int main() {
 *     init_logging("logs/Ceceilia.log");
 *     // TODO: do something
 *     spdlog::drop_all();
 *     return 0;
 * }
 * ```
 */
[[noreturn]] void init_logging(std::string file_path) {
    // sinks
    auto console = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto file_logger = std::make_shared<spdlog::sinks::daily_file_sink_mt>(file_path, 0, 0);

    std::initializer_list<spdlog::sink_ptr> sinks{console, file_logger};
    auto loggers = std::make_shared<spdlog::logger>("logger", sinks);
    loggers->set_level(spdlog::level::debug);
    loggers->set_pattern("%^[%Y-%m-%d %H:%M:%S.%e][%s][line:%#][%l]: %v%$");
    spdlog::register_logger(loggers);
}

class Logger {
private:
    Logger(std::string file_path = "logs/Ceceilia.log") { init_logging(file_path); }
    ~Logger() { spdlog::drop_all(); }

public:
    static std::shared_ptr<spdlog::logger> get_logger() {
        static Logger logger;
        return spdlog::get("logger");
    }
};

auto logger = Logger::get_logger();

#define LOGGER_TRACE(...) SPDLOG_LOGGER_TRACE(logger, __VA_ARGS__)
#define LOGGER_DEBUG(...) SPDLOG_LOGGER_DEBUG(logger, __VA_ARGS__)
#define LOGGER_INFO(...) SPDLOG_LOGGER_INFO(logger, __VA_ARGS__)
#define LOGGER_WARN(...) SPDLOG_LOGGER_WARN(logger, __VA_ARGS__)
#define LOGGER_ERROR(...) SPDLOG_LOGGER_ERROR(logger, __VA_ARGS__)
#define LOGGER_CRITICAL(...) SPDLOG_LOGGER_CRITICAL(logger, __VA_ARGS__)
}  // namespace Ahri

#endif  // !LOGGER_UTILS_HPP
