#pragma once
#ifndef UTILS_H
#define UTILS_H

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#ifdef sleep
#undef sleep
#endif  // !sleep
#ifdef _WIN32
#define sleep(s) Sleep((s) * 1000)
#else
#define sleep(s) sleep(s)
#endif

#endif  // !UTILS_H
