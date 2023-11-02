$root_cache_paths = @(
    ".\build"
    ".\builddir"
    ".\cmake-build-*"
    ".\target"
    ".\out"
    ".\x64"
    ".\Debug"
    ".\.xmake"
)
foreach ($path in $root_cache_paths) {
    if (Test-Path $path) {
        Remove-Item $path -Recurse -Force
    }
}

$recurse_cache_paths = @(
    "x64"
    "Debug"
    "*.exe"
    "*.user"
    # "*.dll"
    # "*.lib"
    # "*.pyd"
    # "*.a"
    # "*.obj"
    "tempCodeRunnerFile.*"
)
foreach ($path in $recurse_cache_paths) {
    Remove-Item * -Include $path -Recurse -Force
}
