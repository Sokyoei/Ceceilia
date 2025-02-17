import std.core;

import hello;
import Ahri;
import Ahri.Sokyoei;

int main(int argc, char const* argv[]) {
    std::cout << say() << std::endl;
    std::cout << Ahri::func(1) << std::endl;
    Ahri::Nono::Animal animal(12, "Nono");
    std::cout << animal << std::endl;
    Ahri::Sokyoei::hello_sokyoei();
    return 0;
}
