// +----------------------+
// | Windows Dump         |
// +--------+-------------+
// | System | Kernel Dump |
// +--------+-------------+
// | User   | Full Dump   |
// |        | Mini Dump   |
// +--------+-------------+

#include <iostream>

#ifdef _WIN32
#include <Windows.h>

#include <DbgHelp.h>
#include <tchar.h>

#pragma comment(lib, "dbghelp.lib")
#elif defined(__linux__)

#else
#error "Dump file are not support"
#endif

LONG CreateMiniDump(EXCEPTION_POINTERS* ExceptionInfo) {
    HANDLE hFile = CreateFile(_T("Dump-Ahri.dmp"), GENERIC_WRITE, FILE_SHARE_WRITE, NULL, CREATE_ALWAYS,
                              FILE_ATTRIBUTE_NORMAL, NULL);

    if ((hFile != NULL) && (hFile != INVALID_HANDLE_VALUE)) {
        MINIDUMP_EXCEPTION_INFORMATION info{GetCurrentThreadId(), ExceptionInfo, FALSE};
        BOOL rv = MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), hFile, MiniDumpNormal,
                                    (ExceptionInfo != 0) ? &info : 0, 0, 0);
        if (!rv) {
            _tprintf(_T("MiniDumpWriteDump faild. Error: %u\n"), GetLastError());
        } else {
            _tprintf(_T("MiniDump created.\n"));
        }
    } else {
        _tprintf(_T("CreateFile faild. Error: %u\n"), GetLastError());
    }
    CloseHandle(&hFile);
    return EXCEPTION_EXECUTE_HANDLER;
}

LONG ExceptionFilter(EXCEPTION_POINTERS* ExceptionInfo) {
    if (true) {
    }
    return CreateMiniDump(ExceptionInfo);
}

void test() {
    SetUnhandledExceptionFilter(ExceptionFilter);
    int* p = nullptr;
    *p = 1;
}

int main(int argc, char* argv[]) {
    test();
    return 0;
}
