#pragma once
#ifndef CECEILIA_TIME_H
#define CECEILIA_TIME_H

#ifdef _WIN32
#include <Windows.h>
#elif defined(__linux__)
#include <unistd.h>
#else
#error "System is not Windows or Linux"
#endif

/**
 * @brief sleep s seconds
 */
#ifdef sleep
#undef sleep
#endif  // !sleep
#ifdef _WIN32
#define sleep(s) Sleep((s) * 1000)
#elif defined(__linux__)
#define sleep(s) sleep(s)
#endif

/**
 * @brief sleep s milliseconds
 */
#ifdef msleep
#undef msleep
#endif  // !msleep
#ifdef _WIN32
#define msleep Sleep(s)
#elif defined(__linux__)
#define msleep usleep((s) * 1000)
#endif

#endif  // !CECEILIA_TIME_H
