#pragma once
#ifndef AHRI_CECEILIA_UTILS_CONSOLE_H
#define AHRI_CECEILIA_UTILS_CONSOLE_H

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ANSI Color Escape
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// clang-format off
/**
 * @brief ANSI Escape character
 */
#define ESCAPE "\033"

/**
 * @brief Foreground color
 */
#define AHRI_FG_BLCAK   ESCAPE "[30m"
#define AHRI_FG_RED     ESCAPE "[31m"
#define AHRI_FG_GREEN   ESCAPE "[32m"
#define AHRI_FG_YELLOW  ESCAPE "[33m"
#define AHRI_FG_BLUE    ESCAPE "[34m"
#define AHRI_FG_PURPLE  ESCAPE "[35m"
#define AHRI_FG_CYAN    ESCAPE "[36m"
#define AHRI_FG_WHITE   ESCAPE "[37m"

/**
 * @brief Background color
 */
#define AHRI_BG_BLCAK   ESCAPE "[40m"
#define AHRI_BG_RED     ESCAPE "[41m"
#define AHRI_BG_GREEN   ESCAPE "[42m"
#define AHRI_BG_YELLOW  ESCAPE "[43m"
#define AHRI_BG_BLUE    ESCAPE "[44m"
#define AHRI_BG_PURPLE  ESCAPE "[45m"
#define AHRI_BG_CYAN    ESCAPE "[46m"
#define AHRI_BG_WHITE   ESCAPE "[47m"

/// @doc
/// # 256 color
/// Foreground color is "\033[38;5;Cm"
/// Background color is "\033[48;5;Cm"
///
/// # 24-bits color, R G B is 0~255
/// Foreground color is "\033[38;2;R;G;Bm"
/// Background color is "\033[48;2;R;G;Bm"
#define AHRI_FG_PINK ESCAPE "[38;2;255;192;203m"
#define AHRI_BG_PINK ESCAPE "[48;2;255;192;203m"

/**
 * @brief Style
 */
#define AHRI_RESET              ESCAPE "[0m"
#define AHRI_BOLD               ESCAPE "[1m"
#define AHRI_DIM                ESCAPE "[2m"
#define AHRI_ITALIC             ESCAPE "[3m"
#define AHRI_SINGLE_UNDERLINE   ESCAPE "[4m"
#define AHRI_FLICKER            ESCAPE "[5m"
#define AHRI_FLICKER2           ESCAPE "[6m"
#define AHRI_REVERSE_COLOR      ESCAPE "[7m"
#define AHRI_YINXING            ESCAPE "[8m"
#define AHRI_STRIKETHROUGH      ESCAPE "[9m"
#define AHRI_OVERLINE           ESCAPE "[21m"
#define AHRI_DOUBLE_UNDERLINE   ESCAPE "[53m"

/**
 * @brief redefine
 */
#define COLOR_BLCAK     AHRI_FG_BLCAK
#define COLOR_RED       AHRI_FG_RED
#define COLOR_GREEN     AHRI_FG_GREEN
#define COLOR_YELLOW    AHRI_FG_YELLOW
#define COLOR_BLUE      AHRI_FG_BLUE
#define COLOR_PURPLE    AHRI_FG_PURPLE
#define COLOR_CYAN      AHRI_FG_CYAN
#define COLOR_WHITE     AHRI_FG_WHITE

#define COLOR_RESET AHRI_RESET
// clang-format on

#endif  // !AHRI_CECEILIA_UTILS_CONSOLE_H
