set_xmakever("2.3.0")

set_project("Ceceilia")
set_version("0.0.1")

add_rules("mode.debug", "mode.release")
set_languages("c17", "c++20")

add_cxflags("-fexec-charset=gbk", { tools = "gcc" })
-- add_cxflags("/source-charset:utf-8", { tools = "msvc" })  -- warning C4819: 该文件包含不能在当前代码页(936)中表示的字符。请将该文件保存为 Unicode 格式以防止数据丢失
add_cxflags("/source-charset:utf-8")
add_cxxflags("/Zc:__cplusplus", { tools = "msvc" })

-- include dir
add_includedirs("$(projectdir)")

-- config.h
set_configdir("$(projectdir)")
add_configfiles("config.h.xmake", {filename = "config.h"})
set_configvar("ROOT", "$(projectdir)")  -- FAIL: Windows 下生成的 config.h 使用 '\'

includes("cc")
-- includes("cuda")
