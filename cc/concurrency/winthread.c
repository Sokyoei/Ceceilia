// Windows Thread
//
// CreateThread()
// _beginthread()/_beginthreadex()

#include <stdio.h>

#include <Windows.h>  // CreateThread()
#include <process.h>  // _beginthread()/_beginthreadex()

DWORD WINAPI ThreadFunc(LPVOID args) {
    printf("CreateThread: hello world\n");
}

void __cdecl ThreadFunc2(void* args) {
    printf("_beginthread: hello world\n");
    _endthread();
}

int main(int argc, char const* argv[]) {
    DWORD thread_id;
    HANDLE h = CreateThread(NULL, 0, ThreadFunc, NULL, 0, &thread_id);
    WaitForSingleObject(h, INFINITE);
    CloseHandle(h);

    HANDLE h2 = (HANDLE)_beginthread(ThreadFunc2, 0, NULL);
    WaitForSingleObject(h2, INFINITE);
    CloseHandle(h2);
    return 0;
}
