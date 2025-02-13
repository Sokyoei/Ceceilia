#pragma once
#ifndef AHRI_HPP
#define AHRI_HPP

#include "Ahri.h"

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
 * @brief C++ attribute
 */
#ifdef __has_cpp_attribute
#if __has_cpp_attribute(noreturn)
#define AHRI_NORETURN [[noreturn]]
#else
#define AHRI_NORETURN
#endif

#if __has_cpp_attribute(deprecated)
#define AHRI_DEPRECATED(msg) [[deprecated(msg)]]
#else
#define AHRI_DEPRECATED(msg)
#endif

#if __has_cpp_attribute(nodiscard)
#define AHRI_NODISCARD(msg) [[nodiscard(msg)]]
#else
#define AHRI_NODISCARD(msg)
#endif

#if __has_cpp_attribute(assume)
#define AHRI_ASSUME(msg) [[assume(msg)]]
#else
#define AHRI_ASSUME(msg)
#endif

#if __has_cpp_attribute(fallthrough)
#define AHRI_FALLTHROUGH [[fallthrough]]
#else
#define AHRI_FALLTHROUGH
#endif

#if __has_cpp_attribute(maybe_unused)
#define AHRI_MAYBE_UNUSED [[maybe_unused]]
#else
#define AHRI_maybe_unused
#endif

#if __has_cpp_attribute(carries_dependency)
#define AHRI_CARRIES_DEPENDENCY [[carries_dependency]]
#else
#define AHRI_CARRIES_DEPENDENCY
#endif

#if __has_cpp_attribute(likely)
#define AHRI_LINKELY [[likely]]
#else
#define AHRI_LINKELY
#endif

#if __has_cpp_attribute(unlikely)
#define AHRI_UNLINKELY [[unlikely]]
#else
#define AHRI_UNLINKELY
#endif

#if __has_cpp_attribute(no_unique_address)
#define AHRI_NO_UNIQUE_ADDRESS [[no_unique_address]]
#else
#define AHRI_NO_UNIQUE_ADDRESS
#endif
#endif  // __has_cpp_attribute

#endif  // !AHRI_HPP
