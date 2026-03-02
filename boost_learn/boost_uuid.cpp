#include <iostream>

#include <boost/uuid.hpp>

int main() {
    boost::uuids::random_generator gen;
    boost::uuids::uuid u = gen();
    std::cout << "Random UUID: " << u << '\n';
    return 0;
}
