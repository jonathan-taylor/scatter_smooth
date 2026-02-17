project('scatter_smooth', 'cpp',
  version : run_command('get-version.py', check: true).stdout().strip(),
  default_options : ['warning_level=3', 'cpp_std=c++14'])

py = import('python').find_installation(pure: false)

# Eigen is vendored in src/eigen
inc = include_directories('src/eigen')

# Pybind11 dependency
pybind11_dep = dependency('pybind11')

# Source files
sources = [
  'src/main.cpp',
  'src/utils.cpp',
  'src/natural_spline.cpp',
  'src/reinsch.cpp',
  'src/bspline.cpp',
  'src/loess.cpp'
]

# Extension module
# This compiles the C++ code into an extension module named '_scatter_smooth_extension'
# and installs it into the 'scatter_smooth' directory.
py.extension_module(
  '_scatter_smooth_extension',
  sources,
  include_directories : inc,
  dependencies : pybind11_dep,
  install : true,
  subdir : 'scatter_smooth'
)

# Install Python sources
# This installs the 'scatter_smooth' python package.
py.install_sources(
    [
        'scatter_smooth/__init__.py',
        'scatter_smooth/datasets.py',
        'scatter_smooth/splines.py',
        'scatter_smooth/loess.py',
    ],
    subdir: 'scatter_smooth'
)
install_subdir(
    'scatter_smooth/data',
    install_dir: py.get_install_dir() / 'scatter_smooth'
)
