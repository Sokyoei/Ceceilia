/**
 * @file logger_utils.hpp
 * @date 2024/08/22
 * @author Sokyoei
 * @details
 * logger
 */

#ifndef AHRI_CECEILIA_UTILS_LOGGER_UTILS_HPP
#define AHRI_CECEILIA_UTILS_LOGGER_UTILS_HPP

#include <initializer_list>
#include <string>

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include <fmt/core.h>
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "Ahri/Ahri.hpp"
#include "console.h"

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
inline void init_logging(std::string file_path) {
    // sinks
    auto console = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto file_logger = std::make_shared<spdlog::sinks::daily_file_sink_mt>(file_path, 0, 0);
    std::string console_pattern = fmt::format("{}[%Y-%m-%d %H:%M:%S.%e]{}{}[%s:%#]{}%^[%l]: %v%$", COLOR_GREEN,
                                              COLOR_RESET, COLOR_CYAN, COLOR_RESET);
    console->set_pattern(console_pattern);
    file_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e][%s:%#]%^[%l]: %v%$");

    std::initializer_list<spdlog::sink_ptr> sinks{console, file_logger};
    auto loggers = std::make_shared<spdlog::logger>("logger", sinks);
    loggers->set_level(spdlog::level::debug);
    spdlog::register_logger(loggers);
}

enum class LogLevel { track, debug, info, warn, error, critical };

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

inline auto logger = Logger::get_logger();

#define AHRI_LOGGER_TRACE(...) SPDLOG_LOGGER_TRACE(Ahri::logger, __VA_ARGS__)
#define AHRI_LOGGER_DEBUG(...) SPDLOG_LOGGER_DEBUG(Ahri::logger, __VA_ARGS__)
#define AHRI_LOGGER_INFO(...) SPDLOG_LOGGER_INFO(Ahri::logger, __VA_ARGS__)
#define AHRI_LOGGER_WARN(...) SPDLOG_LOGGER_WARN(Ahri::logger, __VA_ARGS__)
#define AHRI_LOGGER_ERROR(...) SPDLOG_LOGGER_ERROR(Ahri::logger, __VA_ARGS__)
#define AHRI_LOGGER_CRITICAL(...) SPDLOG_LOGGER_CRITICAL(Ahri::logger, __VA_ARGS__)

// #define LOGGER_TRACE(...) AHRI_LOGGER_TRACE(__VA_ARGS__)
// #define LOGGER_DEBUG(...) AHRI_LOGGER_DEBUG(__VA_ARGS__)
// #define LOGGER_INFO(...) AHRI_LOGGER_INFO(__VA_ARGS__)
// #define LOGGER_WARN(...) AHRI_LOGGER_WARN(__VA_ARGS__)
// #define LOGGER_ERROR(...) AHRI_LOGGER_ERROR(__VA_ARGS__)
// #define LOGGER_CRITICAL(...) AHRI_LOGGER_CRITICAL(__VA_ARGS__)

// #define LOGT(...) LOGGER_TRACE(__VA_ARGS__)
// #define LOGD(...) LOGGER_DEBUG(__VA_ARGS__)
// #define LOGI(...) LOGGER_INFO(__VA_ARGS__)
// #define LOGW(...) LOGGER_WARN(__VA_ARGS__)
// #define LOGE(...) LOGGER_ERROR(__VA_ARGS__)
// #define LOGC(...) LOGGER_CRITICAL(__VA_ARGS__)
}  // namespace Ahri

#endif  // !AHRI_CECEILIA_UTILS_LOGGER_UTILS_HPP
