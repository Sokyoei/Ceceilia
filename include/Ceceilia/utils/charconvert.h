#pragma once
#ifndef CHARCONVERT_H
#define CHARCONVERT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Ahri/Ahri.h"

#ifdef WINDOWS
#include <Windows.h>
#else
#include <unistd.h>
#endif

char* gbk_to_utf8(const char* gbk_str);

#endif  // !CHARCONVERT_H
