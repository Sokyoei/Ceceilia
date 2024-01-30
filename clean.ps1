$root_cache_paths = @(
    ".\.xmake"
    ".\build"
    ".\builddir"
    ".\Debug"
    ".\out"
    ".\target"
    ".\x64"
    ".\cmake-build-*"
)
foreach ($path in $root_cache_paths) {
    if (Test-Path $path) {
        Remove-Item $path -Recurse -Force
    }
}

$recurse_cache_paths = @(
    "Debug"
    "x64"
    "*.exe"
    "*.user"
    # "*.dll"
    # "*.lib"
    # "*.pyd"
    # "*.a"
    "*.obj"
    "*.ilk"
    "*.pdb"
    "tempCodeRunnerFile.*"
)
foreach ($path in $recurse_cache_paths) {
    Remove-Item * -Include $path -Recurse -Force
}
