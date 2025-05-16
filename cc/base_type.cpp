/**
 * C/C++ base type
 */

#include <iostream>
#include <limits>

#include "Ceceilia.hpp"

namespace Ahri {
void int_type() {
    // short
    short short_;
    short int short_int_;
    signed short signed_short_;
    signed short int signed_short_int_;

    unsigned short unsigned_short_;
    unsigned short int unsigned_short_int_;
    // int
    int int_;
    signed signed_;
    signed int signed_int_;

    unsigned unsigned_;
    unsigned int unsigned_int_;
    // long
    long long_;
    long int long_int_;
    signed long signed_long_;
    signed long int signed_long_int_;

    unsigned long unsigned_long_;
    unsigned long int unsigned_long_int_;
#ifdef AHRI_CXX11
    // long long
    long long long_long_;
    long long int long_long_int_;
    signed long long signed_long_long_;
    signed long long int signed_long_long_int_;

    unsigned long long unsigned_long_long_;
    unsigned long long int unsigned_long_long_int_;
#endif  // AHRI_CXX11
}

void float_type() {
    float float_;
    double double_;
    long double long_double_;
#ifdef AHRI_CXXXX
    std::float16_t float16_t_;
    std::float32_t float32_t_;
    std::float64_t float64_t_;
    std::float128_t float128_t_;
    std::bfloat16_t bfloat16_t_;
#endif  // AHRI_CXX23
}

void char_type() {
    char char_;
    wchar_t wchar_t_;
#ifdef AHRI_CXX20
#ifdef __cpp_char8_t
    char8_t char8_t_;
#endif  // __cpp_char8_t
#endif  // AHRI_CXX20
#ifdef AHRI_CXX11
    char16_t char16_t_;
    char32_t char32_t_;
#endif  // AHRI_CXX11
}

bool bool_;
std::nullptr_t nullptr_t_;
std::size_t size_t_;
std::byte byte_;
/**
 * TrivialType(平凡类型)
 * Standard-layout Type(标准布局类型)
 * POD(Plain Old Data)
 */
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    return 0;
}
