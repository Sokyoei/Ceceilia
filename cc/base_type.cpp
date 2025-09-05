/**
 * C/C++ base type
 */

#include "Ahri/Ahri.hpp"

#include <iostream>
#include <limits>

namespace Ahri {
void int_type() {
    // short
    AHRI_MAYBE_UNUSED short short_;
    AHRI_MAYBE_UNUSED short int short_int_;
    AHRI_MAYBE_UNUSED signed short signed_short_;
    AHRI_MAYBE_UNUSED signed short int signed_short_int_;

    AHRI_MAYBE_UNUSED unsigned short unsigned_short_;
    AHRI_MAYBE_UNUSED unsigned short int unsigned_short_int_;
    // int
    AHRI_MAYBE_UNUSED int int_;
    AHRI_MAYBE_UNUSED signed signed_;
    AHRI_MAYBE_UNUSED signed int signed_int_;

    AHRI_MAYBE_UNUSED unsigned unsigned_;
    AHRI_MAYBE_UNUSED unsigned int unsigned_int_;
    // long
    AHRI_MAYBE_UNUSED long long_;
    AHRI_MAYBE_UNUSED long int long_int_;
    AHRI_MAYBE_UNUSED signed long signed_long_;
    AHRI_MAYBE_UNUSED signed long int signed_long_int_;

    AHRI_MAYBE_UNUSED unsigned long unsigned_long_;
    AHRI_MAYBE_UNUSED unsigned long int unsigned_long_int_;
#ifdef AHRI_CXX11
    // long long
    AHRI_MAYBE_UNUSED long long long_long_;
    AHRI_MAYBE_UNUSED long long int long_long_int_;
    AHRI_MAYBE_UNUSED signed long long signed_long_long_;
    AHRI_MAYBE_UNUSED signed long long int signed_long_long_int_;

    AHRI_MAYBE_UNUSED unsigned long long unsigned_long_long_;
    AHRI_MAYBE_UNUSED unsigned long long int unsigned_long_long_int_;
#endif  // AHRI_CXX11
}

void float_type() {
    AHRI_MAYBE_UNUSED float float_;
    AHRI_MAYBE_UNUSED double double_;
    AHRI_MAYBE_UNUSED long double long_double_;
#ifdef AHRI_CXXXX
    AHRI_MAYBE_UNUSED std::float16_t float16_t_;
    AHRI_MAYBE_UNUSED std::float32_t float32_t_;
    AHRI_MAYBE_UNUSED std::float64_t float64_t_;
    AHRI_MAYBE_UNUSED std::float128_t float128_t_;
    AHRI_MAYBE_UNUSED std::bfloat16_t bfloat16_t_;
#endif  // AHRI_CXX23
}

void char_type() {
    AHRI_MAYBE_UNUSED char char_;
    AHRI_MAYBE_UNUSED wchar_t wchar_t_;
#ifdef AHRI_CXX20
#ifdef __cpp_char8_t
    AHRI_MAYBE_UNUSED char8_t char8_t_;
#endif  // __cpp_char8_t
#endif  // AHRI_CXX20
#ifdef AHRI_CXX11
    AHRI_MAYBE_UNUSED char16_t char16_t_;
    AHRI_MAYBE_UNUSED char32_t char32_t_;
#endif  // AHRI_CXX11
}

AHRI_MAYBE_UNUSED bool bool_;
AHRI_MAYBE_UNUSED std::nullptr_t nullptr_t_;
AHRI_MAYBE_UNUSED std::size_t size_t_;
AHRI_MAYBE_UNUSED std::byte byte_;
/**
 * TrivialType(平凡类型)
 * Standard-layout Type(标准布局类型)
 * POD(Plain Old Data)
 */
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    return 0;
}
