project(
    'Ceceilia',
    'c',
    'cpp',
    version: '0.0.1',
    license: 'MIT',
    default_options: ['c_std=c17', 'cpp_std=c++20'],
)

cxx = meson.get_compiler('cpp')
cc = meson.get_compiler('c')
has_nvcc = add_languages('cuda', required: false)
if has_nvcc
    nvcc = meson.get_compiler('cuda')
endif

ROOT = '"' + meson.current_source_dir().replace('\\', '/') + '"'
include_root = include_directories('.')
include = include_directories('include')

# library
fmt = dependency('fmt')
spdlog = dependency('spdlog')
tinyxml2 = dependency('tinyxml2')
nlohmann_json = dependency('nlohmann_json')
yaml_cpp = dependency('yaml-cpp')
tomlplusplus = dependency('tomlplusplus', method: 'pkg-config')
gtest = dependency('gtest')
gtest_main = dependency('gtest_main')

subdir('cc')
subdir('tests')

if has_nvcc
    subdir('cuda')
endif
if get_option('boost_learn')
    subdir('boost_learn')
endif
if get_option('folly_learn')
    subdir('folly_learn')
endif
if get_option('protobuf_learn')
    subdir('protobuf_learn')
endif

# config.h
cfg_data = configuration_data()
cfg_data.set('ROOT', ROOT)
cfg_data.set('USE_FMT', fmt.found())
cfg_data.set('USE_SPDLOG', spdlog.found())
cfg_data.set('USE_TINYXML2', tinyxml2.found())
cfg_data.set('USE_NLOHMANN_JSON', nlohmann_json.found())
cfg_data.set('USE_YAML_CPP', yaml_cpp.found())
cfg_data.set('USE_TOMLPLUSPLUS', tomlplusplus.found())
cfg_data.set('USE_GTEST', gtest.found())
configure_file(
    input: meson.current_source_dir() / 'config.h.meson',
    output: 'config.h', # in build dir
    configuration: cfg_data,
)
