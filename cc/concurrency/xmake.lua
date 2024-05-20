target("actor_csp")
    set_kind("binary")
    add_files("actor_csp.cpp")
target_end()

target("atomic")
    set_kind("binary")
    add_files("atomic.cpp")
target_end()

target("condition_variable")
    set_kind("binary")
    add_files("condition_variable.cpp")
target_end()

target("future")
    set_kind("binary")
    add_files("future.cpp")
target_end()

target("lock")
    set_kind("binary")
    add_files("lock.cpp")
target_end()

target("singleton")
    set_kind("binary")
    add_files("singleton.cpp")
target_end()

target("thread")
    set_kind("binary")
    add_files("thread.cpp")
target_end()

target("threadpool_example")
    set_kind("binary")
    add_files("threadpool_example.cpp")
target_end()

-- if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
--     find_package(Threads REQUIRED)
--     if(Threads_FOUND)
--         add_executable(pthread pthread.c)
--         target_link_libraries(pthread PRIVATE Threads::Threads)
--         add_executable(pthreadpool_example pthreadpool_example.c pthreadpool.h pthreadpool.c)
--         target_link_libraries(pthreadpool_example PRIVATE Threads::Threads)
--     endif()
-- endif()

if is_os("windows") then
    target("winthread")
        set_kind("binary")
        add_files("winthread.c")
    target_end()

    target("winthreadpool")
        set_kind("binary")
        add_files("winthreadpool.c")
    target_end()
end
