#pragma once
#ifndef AHRI_HPP
#define AHRI_HPP

#include "Ahri.h"

namespace Ahri {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// C++ Standard
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief C++ standard
 */
#ifdef __cplusplus
#ifdef _MSC_VER
#define AHRI_CPLUSPLUS _MSVC_LANG
#else
#define AHRI_CPLUSPLUS __cplusplus
#endif

#define AHRI_CXX 1
#if AHRI_CPLUSPLUS >= 199711L
#define AHRI_CXX98 1
#if AHRI_CPLUSPLUS >= 201103L
#define AHRI_CXX11 1
#if AHRI_CPLUSPLUS >= 201402L
#define AHRI_CXX14 1
#if AHRI_CPLUSPLUS >= 201703L
#define AHRI_CXX17 1
#if AHRI_CPLUSPLUS >= 202002L
#define AHRI_CXX20 1
#if AHRI_CPLUSPLUS >= 202302L
#define AHRI_CXX23 1
#if AHRI_CPLUSPLUS >= 202612L  // not publish
#define AHRI_CXX26 1
#endif  // C++26
#endif  // C++23
#endif  // C++20
#endif  // C++17
#endif  // C++14
#endif  // C++11
#endif  // C++98
#endif  // __cplusplus

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// C++ Keyword/Attribute
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief C++20 module/export
 */
#if defined(AHRI_CXX20) && defined(__cpp_modules)
#define AHRI_CXX20_MODULE(name) module name;
#define AHRI_CXX20_EXPORT_MODULE(name) export module name;
#define AHRI_CXX20_GLOBAL_MODULE module;
#define AHRI_CXX20_EXPORT export
#else
#define AHRI_CXX20_MODULE(name)
#define AHRI_CXX20_EXPORT_MODULE(name)
#define AHRI_CXX20_GLOBAL_MODULE
#define AHRI_CXX20_EXPORT AHRI_API
#endif

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

#if __has_cpp_attribute(indeterminate)
#define AHRI_INDETERMINATE [[indeterminate]]
#else
#define AHRI_INDETERMINATE
#endif
#endif  // __has_cpp_attribute
}  // namespace Ahri

#endif  // !AHRI_HPP
