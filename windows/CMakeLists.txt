add_executable(winapi winapi.cpp)
add_executable(winmain winmain.cpp)
target_link_options(winmain PRIVATE "/SUBSYSTEM:WINDOWS")
