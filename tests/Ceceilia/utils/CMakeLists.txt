add_executable(test_error_utils test_error_utils.c)
target_link_libraries(test_error_utils ${GTEST_BOTH_LIBRARIES})

add_executable(test_json test_json.cpp)

if(USE_MOSQUITTO)
    add_executable(test_mqtt_publish test_mqtt_publish.cpp)
    target_include_directories(test_mqtt_publish PRIVATE ${MOSQUITTO_INCLUDE_DIRS})
    target_link_libraries(test_mqtt_publish PRIVATE ${MOSQUITTO_LIBRARIES} fmt::fmt-header-only)

    add_executable(test_mqtt_subscribe test_mqtt_subscribe.cpp)
    target_include_directories(test_mqtt_subscribe PRIVATE ${MOSQUITTO_INCLUDE_DIRS})
    target_link_libraries(test_mqtt_subscribe PRIVATE ${MOSQUITTO_LIBRARIES} fmt::fmt-header-only)
endif(USE_MOSQUITTO)

add_executable(test_logger_utils test_logger_utils.cpp)
target_link_libraries(test_logger_utils PRIVATE spdlog::spdlog_header_only)
