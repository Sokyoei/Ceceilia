add_executable(download download.c)

add_executable(IOCP_server IOCP_server.cpp)
add_executable(winapi winapi.cpp)
add_executable(winmain winmain.cpp)
target_link_options(winmain PRIVATE "/SUBSYSTEM:WINDOWS")
