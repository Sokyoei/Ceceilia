# type: ignore
from conan import ConanFile
from conan.tools.cmake import cmake_layout, CMake, CMakeToolchain
from conan.tools.microsoft import is_msvc
from conan.errors import ConanInvalidConfiguration


class Ceceilia(ConanFile):
    name = "Ceceilia"
    version = "0.0.1"
    license = "MIT"
    author = "Sokyoei"
    url = "https://github.com/Sokyoei/Ceceilia"
    description = "This is Sokyoei's C/C++/CUDA tutorials and utils project"
    topics = ("distributed", "system", "library")

    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def validate(self):
        if self.settings.get_safe("compiler.cppstd"):
            cppstd = int(str(self.settings.compiler.cppstd))
            if cppstd < 17:
                raise ConanInvalidConfiguration("C++17 or higher is required")

    def configure(self) -> None:
        if self.settings.get_safe("compiler.cppstd") is None:
            self.settings.compiler.cppstd = "20"

    def build(self) -> None:
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def layout(self) -> None:
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["CMAKE_CXX_STANDARD"] = "20"
        tc.variables["CMAKE_CXX_STANDARD_REQUIRED"] = "ON"
        tc.generate()

    def package(self) -> None:
        cmake = CMake(self)
        cmake.install()

    def requirements(self) -> None:
        self.requires("fmt/10.2.1")
        self.requires("spdlog/1.14.1")
        self.requires("gtest/1.16.0")
        self.requires("folly/2024.08.12.00")
