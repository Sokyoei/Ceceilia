/**
 * @file error_utils.h
 * @date 2024/09/18
 * @author Sokyoei
 *
 *
 */

#pragma once
#ifndef AHRI_CECEILIA_UTILS_ERROR_UTILS_H
#define AHRI_CECEILIA_UTILS_ERROR_UTILS_H

#include <setjmp.h>

/**
 * @example
 *
 * ```c
 * #define AHRI_ERROR (1)
 * #define SOKYOEI_ERROR (2)
 * #define NONO_ERROR (3)
 *
 * TRY {
 *     // TODO
 *     THROW(SOKYOEI_ERROR);
 * }
 * CATCH(AHRI_ERROR) {
 *     // TODO
 * }
 * CATCH(SOKYOEI_ERROR) {
 *     // TODO
 * }
 * CATCH(NONO_ERROR) {
 *     // TODO
 * }
 * ENDTRY
 * ```
 */
#define TRY                    \
    do {                       \
        jmp_buf env;           \
        switch (setjmp(env)) { \
            case 0:

#define CATCH(x) \
    break;       \
    case x:

#define THROW(x) longjmp(env, x)

#define ENDTRY \
    }          \
    }          \
    while (0)  \
        ;

#endif  // !AHRI_CECEILIA_UTILS_ERROR_UTILS_H
