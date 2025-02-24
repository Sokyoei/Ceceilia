project(
    'Ceceilia',
    'c',
    'cpp',
    version: '0.0.1',
    license: 'MIT',
)

cxx = meson.get_compiler('cpp')
cc = meson.get_compiler('c')
has_nvcc = add_languages('cuda', required: false)
if has_nvcc
    nvcc = meson.get_compiler('cuda')
endif

ROOT = '"' + meson.current_source_dir().replace('\\', '/') + '"'
config_h = include_directories('.')

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

if get_option('cuda') and has_nvcc
    subdir('cuda')
endif
if get_option('asio_learn')
    subdir('asio_learn')
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
