#include <exception>
#include <iostream>

#include "Ahri/Ceceilia/utils/mqtt_utils.hpp"

#define BROKER "broker.emqx.io"
#define PORT 1883
#define TOPIC "/cpp/mqtt"
#define CLIENT_ID "subscribe_client"

int main(int argc, char const* argv[]) {
    try {
        auto client = Ahri::MQTTClient(CLIENT_ID, BROKER, PORT);
        client.subscribe(nullptr, TOPIC, 0);
        client.loop_forever();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
    }
    return 0;
}
