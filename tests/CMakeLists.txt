add_executable(test_Ahri test_Ahri.cpp)
target_link_libraries(test_Ahri ${GTEST_BOTH_LIBRARIES})

add_executable(test_mqtt_publish test_mqtt_publish.cpp)
target_include_directories(test_mqtt_publish PRIVATE ${MOSQUITTO_INCLUDE_DIRS})
target_link_libraries(test_mqtt_publish PRIVATE ${MOSQUITTO_LIBRARIES} fmt::fmt-header-only)

add_executable(test_mqtt_subscribe test_mqtt_subscribe.cpp)
target_include_directories(test_mqtt_subscribe PRIVATE ${MOSQUITTO_INCLUDE_DIRS})
target_link_libraries(test_mqtt_subscribe PRIVATE ${MOSQUITTO_LIBRARIES} fmt::fmt-header-only)

add_executable(test_logger_utils test_logger_utils.cpp)
target_link_libraries(test_logger_utils PRIVATE spdlog::spdlog_header_only)
