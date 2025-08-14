if is_plat("windows") then
    target("client_example")
        set_kind("binary")
        add_files("client_example.cpp")
    target_end()
end

if is_plat("windows") then
    target("server_example")
        set_kind("binary")
        add_files("server_example.cpp")
    target_end()
end
