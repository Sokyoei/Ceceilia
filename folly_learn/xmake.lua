target("folly_executor")
    set_kind("binary")
    add_files("folly_executor.cpp")
    add_packages("folly")
    add_packages("fmt")
target_end()

target("folly_string")
    set_kind("binary")
    add_files("folly_string.cpp")
    add_packages("folly")
    add_packages("fmt")
target_end()
