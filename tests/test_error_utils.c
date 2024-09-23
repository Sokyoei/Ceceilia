#include <stdio.h>

#include "Ceceilia/utils/error_utils.h"

#define AHRI_ERROR (1)
#define SOKYOEI_ERROR (2)
#define NONO_ERROR (3)

int main(int argc, char const* argv[]) {
    TRY {
        THROW(AHRI_ERROR);
    }
    CATCH(AHRI_ERROR) {
        printf("AHRI_ERROR");
    }
    CATCH(SOKYOEI_ERROR) {
        printf("SOKYOEI_ERROR");
    }
    CATCH(NONO_ERROR) {
        printf("NONO_ERROR");
    }
    ENDTRY
    return 0;
}
