/**
 * @file winthreadpool.c
 * @date 2023/12/12
 * @author Sokyoei
 * @details
 * Windows ThreadPool example
 */

#include <stdio.h>

#ifdef _WIN32
#include <Windows.h>
#include <process.h>
#else
#error "OS is not Windows"
#endif

VOID NTAPI simple_callback(PTP_CALLBACK_INSTANCE pci, PVOID pv) {}
BOOL try_submit_threadpool_callback(PTP_SIMPLE_CALLBACK psc, PVOID pv, PTP_CALLBACK_ENVIRON pce) {}

int main(int argc, char const* argv[]) {
    PTP_POOL tp = CreateThreadpool(NULL);
    SetThreadpoolThreadMaximum(tp, 16);
    SetThreadpoolThreadMinimum(tp, 4);
    // SetThreadpoolCallbackPool();
    // InitializeThreadpoolEnvironment();
    CloseThreadpool(tp);
    return 0;
}
