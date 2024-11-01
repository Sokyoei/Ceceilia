#include <fstream>
#include <iostream>
#include <string>

#include <fmt/core.h>

#include "protobuf_v2_learn.pb.h"

int main(int argc, char const* argv[]) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    Ahri::Person person;
    person.set_id(1);
    person.set_name("Ahri");

    std::string str;
    person.SerializeToString(&str);
    fmt::println("{}", str);

    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
