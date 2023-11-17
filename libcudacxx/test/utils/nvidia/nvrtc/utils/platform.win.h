#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <string>

static void platform_exec(char const* process, char ** args, size_t nargs) {
    std::string cl{};

    STARTUPINFOA si{};
    PROCESS_INFORMATION pi{};

    si.cb = sizeof(si);

    cl.append(process);

    for (auto iter = args; iter < (args + nargs); iter++) {
        cl.append(" ");
        cl.append(*iter);
    }

    printf("Running command: %s\r\n", cl.data());

    bool exec_result = CreateProcess(
        NULL,
        (LPSTR)cl.data(),
        NULL,
        NULL,
        false,
        false,
        NULL,
        NULL,
        &si,
        &pi
    );

    if (!exec_result) {
        printf("Launch error: %i",  GetLastError());
    }

    WaitForSingleObject(pi.hProcess, INFINITE);

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    ExitProcess(0);
}
