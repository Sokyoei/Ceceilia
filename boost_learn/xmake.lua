target("boost_threadpool")
    set_kind("binary")
    add_files("boost_threadpool.cpp")
    add_packages("boost")
target_end()

target("boost_uuid")
    set_kind("binary")
    add_files("boost_uuid.cpp")
    add_packages("boost")
target_end()
