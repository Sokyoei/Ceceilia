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
message(has_nvcc)
if has_nvcc
    nvcc = meson.get_compiler('cuda')
endif

ROOT = '"' + meson.current_source_dir().replace('\\', '/') + '"'
config_h = include_directories('.')

subdir('cc')

if get_option('cuda') and has_nvcc
    subdir('cuda')
endif
if get_option('asio_learn')
    subdir('asio_learn')
endif

# config.h
cfg_data = configuration_data()
cfg_data.set('ROOT', ROOT)
configure_file(
    input: meson.current_source_dir() / 'config.h.meson',
    output: 'config.h', # in build dir
    configuration: cfg_data,
)
