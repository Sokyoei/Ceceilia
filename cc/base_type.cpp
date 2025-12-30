/**
 * @file base_type.cpp
 * @date 2025/11/04
 * @author Sokyoei
 * @details
 * C/C++ base type
 */

#include "Ahri/Ahri.hpp"

#include <math.h>

#include <iostream>
#include <limits>

namespace Ahri {
void int_type() {
    // short
    AHRI_MAYBE_UNUSED short short_ = 0;
    AHRI_MAYBE_UNUSED short int short_int_ = 0;
    AHRI_MAYBE_UNUSED signed short signed_short_ = 0;
    AHRI_MAYBE_UNUSED signed short int signed_short_int_ = 0;

    AHRI_MAYBE_UNUSED unsigned short unsigned_short_ = 0;
    AHRI_MAYBE_UNUSED unsigned short int unsigned_short_int_ = 0;
    // int
    AHRI_MAYBE_UNUSED int int_ = 0;
    AHRI_MAYBE_UNUSED signed signed_ = 0;
    AHRI_MAYBE_UNUSED signed int signed_int_ = 0;

    AHRI_MAYBE_UNUSED unsigned unsigned_ = 0;
    AHRI_MAYBE_UNUSED unsigned int unsigned_int_ = 0;
    // long
    AHRI_MAYBE_UNUSED long long_ = 0;
    AHRI_MAYBE_UNUSED long int long_int_ = 0;
    AHRI_MAYBE_UNUSED signed long signed_long_ = 0;
    AHRI_MAYBE_UNUSED signed long int signed_long_int_ = 0;

    AHRI_MAYBE_UNUSED unsigned long unsigned_long_ = 0;
    AHRI_MAYBE_UNUSED unsigned long int unsigned_long_int_ = 0;
#ifdef AHRI_CXX11
    // long long
    AHRI_MAYBE_UNUSED long long long_long_ = 0;
    AHRI_MAYBE_UNUSED long long int long_long_int_ = 0;
    AHRI_MAYBE_UNUSED signed long long signed_long_long_ = 0;
    AHRI_MAYBE_UNUSED signed long long int signed_long_long_int_ = 0;

    AHRI_MAYBE_UNUSED unsigned long long unsigned_long_long_ = 0;
    AHRI_MAYBE_UNUSED unsigned long long int unsigned_long_long_int_ = 0;
#endif  // AHRI_CXX11
}

void float_type() {
    AHRI_MAYBE_UNUSED float float_ = NAN;
    AHRI_MAYBE_UNUSED double double_ = NAN;
    AHRI_MAYBE_UNUSED long double long_double_ = NAN;
#ifdef AHRI_CXXXX
    AHRI_MAYBE_UNUSED std::float16_t float16_t_;
    AHRI_MAYBE_UNUSED std::float32_t float32_t_;
    AHRI_MAYBE_UNUSED std::float64_t float64_t_;
    AHRI_MAYBE_UNUSED std::float128_t float128_t_;
    AHRI_MAYBE_UNUSED std::bfloat16_t bfloat16_t_;
#endif  // AHRI_CXX23
}

void char_type() {
    AHRI_MAYBE_UNUSED char char_ = 0;
    AHRI_MAYBE_UNUSED wchar_t wchar_t_ = 0;
#ifdef AHRI_CXX20
#ifdef __cpp_char8_t
    AHRI_MAYBE_UNUSED char8_t char8_t_ = 0;
#endif  // __cpp_char8_t
#endif  // AHRI_CXX20
#ifdef AHRI_CXX11
    AHRI_MAYBE_UNUSED char16_t char16_t_ = 0;
    AHRI_MAYBE_UNUSED char32_t char32_t_ = 0;
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
/// @brief TrivialType and Non-TrivialType 平凡类型和非平凡类型』
/// @details 平凡与非平凡的核心在于是否符合编译器默认生成的最基础行为
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    return 0;
}
