/**
 * @file mqtt_utils.hpp
 * @date 2024/08/22
 * @author Sokyoei
 *
 *
 */

#ifndef AHRI_CECEILIA_UTILS_MQTT_UTILS_HPP
#define AHRI_CECEILIA_UTILS_MQTT_UTILS_HPP

#include <fmt/core.h>
#include <mosquittopp.h>

namespace Ahri {
class MQTTClient : public mosqpp::mosquittopp {
private:
public:
    MQTTClient(const char* id, const char* host, int port = 1883, int keepalive = 60) : mosquittopp(id) {
        mosqpp::lib_init();
        connect(host, port, keepalive);
    }

    MQTTClient(const char* id,
               const char* username,
               const char* password,
               const char* host,
               int port = 1883,
               int keepalive = 60)
        : mosquittopp(id) {
        mosqpp::lib_init();
        username_pw_set(username, password);
        connect(host, port, keepalive);
    }

    ~MQTTClient() {
        loop_stop();
        disconnect();
        mosqpp::lib_cleanup();
    }

    void on_connect(int rc) override {
        if (rc == MOSQ_ERR_SUCCESS) {
            fmt::println("Connected success.");
        } else {
            fmt::println("Connected fail, error is {}.", mosqpp::strerror(rc));
        }
    }

    void on_subscribe(int mid, int qos_count, const int* granted_qos) override {
        fmt::println("Subscription mid {}.", mid);
    }

    void on_message(const struct mosquitto_message* message) override {
        // TODO: Callback function for process message
        fmt::println("Received {} from {}", static_cast<char*>(message->payload), message->topic);
    }
};
}  // namespace Ahri

#endif  // !AHRI_CECEILIA_UTILS_MQTT_UTILS_HPP
