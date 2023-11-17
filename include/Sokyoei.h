#ifndef SOKYOEI_H
#define SOKYOEI_H

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Language standard
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief C standard
 */
#ifdef __STDC__
#define C 1
#if __STDC_VERSION__ >= 199901L
#define C99 1
#if __STDC_VERSION__ >= 201112L
#define C11 1
#if __STDC_VERSION__ >= 201710L
#define C17 1
#if __STDC_VERSION__ >= 202000L
#define C23 1
#endif  // C23
#endif  // C17
#endif  // C11
#endif  // C99
#endif  // __STDC__

/**
 * @brief C++ standard
 */
#ifdef __cplusplus
#define CXX 1
#if __cplusplus >= 199711L || _MSVC_LANG >= 199711L
#define CXX98 1
#if __cplusplus >= 201103L || _MSVC_LANG >= 201103L
#define CXX11 1
#if __cplusplus >= 201402L || _MSVC_LANG >= 201402L
#define CXX14 1
#if __cplusplus >= 201703L || _MSVC_LANG >= 201703L
#define CXX17 1
#if __cplusplus >= 202002L || _MSVC_LANG >= 202002L
#define CXX20 1
#if __cplusplus >= 202302L || _MSVC_LANG >= 202302L
#define CXX23 1
#endif  // C++23
#endif  // C++20
#endif  // C++17
#endif  // C++14
#endif  // C++11
#endif  // C++98
#endif  // __cplusplus

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Export
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define AHRI_EXPORT

#ifdef _MSC_VER  // MSVC
#ifdef AHRI_EXPORT
#define AHRI_API __declspec(dllexport)
#else
#define AHRI_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__GNUG__) && __GNUC__ > 4  // GCC
#define AHRI_API __attribute__((visibility("default")))
#else
#define AHRI_API
#endif

#endif  // !SOKYOEI_H
