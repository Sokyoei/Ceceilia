set_xmakever("2.3.0")

set_project("Ceceilia")
set_version("0.0.1")

add_rules("mode.debug", "mode.release")
set_languages("c17", "c++20")

-- options
option("BUILD_CUDA")
    set_default(false)
    set_showmenu(true)
    set_description("build cuda")
option_end()

option("BUILD_asio_learn")
    set_default(false)
    set_showmenu(true)
    set_description("build asio_learn subproject")
option_end()

add_defines("AHRI_EXPORT"--[[ , "_DEBUG" ]])

set_warnings("all")

-- GCC
add_cxflags("-fdiagnostics-color=always", { tools = "gcc" })
if is_os("windows") then
end
add_cxflags("-fexec-charset=gbk", { tools = "gcc" })
-- MSVC
add_cxflags("/EHsc", "/source-charset:utf-8", { tools = "cl" })
add_cflags("/Zc:__STDC__", { tools = "cl" })
add_cxxflags("/Zc:__cplusplus", { tools = "cl" })
add_ldflags("/subsystem:console")

-- include dir
add_includedirs("$(projectdir)")
add_includedirs("$(projectdir)/include")

-- config.h
set_configdir("$(projectdir)")
add_configfiles("config.h.xmake", { filename = "config.h" })
set_configvar("ROOT", "$(projectdir)") -- FAIL: Windows 下生成的 config.h 使用 '\'

add_requires("gtest", { config = { main = true, shared = true, gmock = true }})
add_requires("fmt", { config = { header_only = true }})
add_requires("nlohmann_json")
add_requires("tinyxml2")
-- add_requires("toml11")
add_requires("toml++")
add_requires("yaml-cpp")

includes("cc")
if option("BUILD_CUDA") then
    includes("cuda")
end
if option("BUILD_asio_learn") then
    includes("asio_learn")
end
includes("tests")
