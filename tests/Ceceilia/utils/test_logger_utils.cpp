#include "Ceceilia/utils/logger_utils.hpp"

int main(int argc, char const* argv[]) {
#ifdef _WIN32
    std::system("chcp 65001");
#endif
    auto logger = Ahri::Logger::get_logger();
    logger->debug("debug 测试");
    logger->info("info 信息");
    logger->warn("warn 警告");
    logger->error("error 错误");
    // spdlog 使用宏可以输出行号和文件名等
    AHRI_LOGGER_DEBUG("debug 测试");
    AHRI_LOGGER_INFO("info 信息");
    AHRI_LOGGER_WARN("warn 警告");
    AHRI_LOGGER_ERROR("error 错误");
    return 0;
}
