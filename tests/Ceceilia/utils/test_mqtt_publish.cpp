#include <chrono>
#include <exception>
#include <iostream>
#include <string>
#include <thread>

#include "Ceceilia/utils/mqtt_utils.hpp"

#define BROKER "broker.emqx.io"
#define PORT 1883
#define TOPIC "/cpp/mqtt"
#define CLIENT_ID "publish_client"

using std::chrono_literals::operator"" s;

int main(int argc, char const* argv[]) {
    try {
        auto client = Ahri::MQTTClient(CLIENT_ID, BROKER, PORT);
        int value = 0;
        while (true) {
            auto message = std::to_string(value);
            client.publish(nullptr, TOPIC, message.size(), message.c_str());
            fmt::println("Send {} to {}.", message, TOPIC);
            value++;
            std::this_thread::sleep_for(1s);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
    }
    return 0;
}
