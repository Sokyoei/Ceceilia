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
 * @brief platform define
 */
#if defined(WIN32) || defined(_WIN32) || defined(_WIN32_) || defined(__WIN32__) || defined(WIN64) || \
    defined(_WIN64) || defined(_WIN64_) || defined(__WIN64__)
#define WINDOWS
#elif defined(__linux__)
#define LINUX
#endif

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
