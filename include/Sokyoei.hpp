#pragma once
#ifndef SOKYOEI_HPP
#define SOKYOEI_HPP

#include "Sokyoei.h"

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

#endif  // !SOKYOEI_HPP
