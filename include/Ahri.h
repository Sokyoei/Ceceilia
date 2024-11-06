#pragma once
#ifndef AHRI_H
#define AHRI_H

/**
 * @brief C standard
 */
#ifdef __STDC__
#define AHRI_C 1
#if __STDC_VERSION__ >= 199901L
#define AHRI_C99 1
#if __STDC_VERSION__ >= 201112L
#define AHRI_C11 1
#if __STDC_VERSION__ >= 201710L
#define AHRI_C17 1
#if __STDC_VERSION__ >= 202000L
#define AHRI_C23 1
#endif  // C23
#endif  // C17
#endif  // C11
#endif  // C99
#endif  // __STDC__

/**
 * @brief C++ standard
 */
#ifdef __cplusplus
#define AHRI_CXX 1
#if __cplusplus >= 199711L || _MSVC_LANG >= 199711L
#define AHRI_CXX98 1
#if __cplusplus >= 201103L || _MSVC_LANG >= 201103L
#define AHRI_CXX11 1
#if __cplusplus >= 201402L || _MSVC_LANG >= 201402L
#define AHRI_CXX14 1
#if __cplusplus >= 201703L || _MSVC_LANG >= 201703L
#define AHRI_CXX17 1
#if __cplusplus >= 202002L || _MSVC_LANG >= 202002L
#define AHRI_CXX20 1
#if __cplusplus >= 202302L || _MSVC_LANG >= 202302L
#define AHRI_CXX23 1
#endif  // C++23
#endif  // C++20
#endif  // C++17
#endif  // C++14
#endif  // C++11
#endif  // C++98
#endif  // __cplusplus

/**
 * @brief export
 */
#ifdef _MSC_VER
#ifdef AHRI_EXPORT
#define AHRI_API __declspec(dllexport)
#else
#define AHRI_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__GNUG__) && __GNUC__ > 4
#define AHRI_API __attribute__((visibility("default")))
#else
#define AHRI_API
#endif

/**
 * @brief assert
 * HACK: not test
 */
#ifndef AHRI_ASSERT
#ifdef _DEBUG
#ifdef _MSC_VER
#define AHRI_ASSERT(x)         \
    do {                       \
        if (!((void)0, (x))) { \
            __debugbreak();    \
        }                      \
    } while (false)
#elif defined()
#define AHRI_ASSERT
#else
#include <assert.h>
#define AHRI_ASSERT assert
#endif
#else  // _DEBUG / no _DEBUG
#define AHRI_ASSERT(x) \
    do {               \
        x              \
    } while (0)
#endif  // _DEBUG
#endif  // !AHRI_ASSERT

#endif  // !AHRI_H
