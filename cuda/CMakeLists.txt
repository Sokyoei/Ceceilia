add_executable(device_info device_info.cu)
target_link_libraries(device_info PRIVATE fmt::fmt-header-only)
set_property(TARGET device_info PROPERTY CUDA_STANDARD 17)
set_property(TARGET device_info PROPERTY CUDA_STANDARD_REQUIRED ON)
# CUDA with Clang on Ubuntu 22.04 build fails with FMT_USE_BITINT=1
if(LINUX)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_definitions(device_info PRIVATE FMT_USE_BITINT=0)
    endif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
endif(LINUX)



add_executable(event event.cu)
add_executable(kernel kernel.cu)
add_executable(stream stream.cu)
add_executable(shared_memory shared_memory.cu)
