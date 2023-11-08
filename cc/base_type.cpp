/**
 * C/C++ base type
 */

#include <iostream>
#include <limits>

namespace Ahri {
// int
short s;
int i;

// float
float f;
double d;
long double ld;

// char
char c;
// char8_t c8;
#if __cplusplus >= 201103L
char16_t c16;
char32_t c32;
#endif  // __cplusplus >= 201103L

std::nullptr_t nil;

}  // namespace Ahri

int main(int argc, char const* argv[]) {
    return 0;
}
