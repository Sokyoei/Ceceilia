/**
 * @file file.cpp
 * @date 2024/01/30
 * @author Sokyoei
 * @details
 * C++ file
 */

#include <fstream>
#include <iostream>
#include <string>

namespace Ahri {
void write_file(std::string file_name) {
    std::ofstream wf;
    wf.open(file_name);
    if (wf.is_open()) {
        wf << "Hello World!" << '\n';
        wf << "Dear Ahri." << '\n';
    } else {
        std::cout << "Failed to open this file." << '\n';
    }
    wf.close();
}

void read_file(std::string file_name) {
    std::ifstream rf(file_name);
    if (rf.is_open()) {
        std::string line;
        while (std::getline(rf, line)) {
            std::cout << line << '\n';
        }
    } else {
        std::cout << "Failed to open this file." << '\n';
    }
    rf.close();
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    Ahri::write_file("Ahri.log");
    Ahri::read_file("Ahri.log");
    return 0;
}
