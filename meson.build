project('smoothing_spline', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3', 'cpp_std=c++14'])

py = import('python').find_installation(pure: false)

# Eigen is vendored in src/eigen
inc = include_directories('src/eigen')

# Pybind11 dependency
pybind11_dep = dependency('pybind11', required : false)
if not pybind11_dep.found()
  # Fallback: try to find it via python command if pkg-config fails
  pybind11_incdir = run_command(py, ['-c', 'import pybind11; print(pybind11.get_include())'], check : true).stdout().strip()
  pybind11_dep = declare_dependency(include_directories : include_directories(pybind11_incdir))
endif

# Source files
sources = [
  'src/main.cpp',
  'src/utils.cpp',
  'src/natural_spline.cpp',
  'src/reinsch.cpp',
  'src/bspline.cpp'
]

# Extension module
# This compiles the C++ code into an extension module named '_spline_extension'
# and installs it into the 'smoothing_spline' directory.
py.extension_module(
  '_spline_extension',
  sources,
  include_directories : inc,
  dependencies : pybind11_dep,
  install : true,
  subdir : 'smoothing_spline'
)

# Install Python sources
# This installs the 'smoothing_spline' python package.
py.install_sources(
    [
        'smoothing_spline/__init__.py',
        'smoothing_spline/datasets.py',
        'smoothing_spline/fitter.py',
    ],
    subdir: 'smoothing_spline'
)
install_subdir(
    'smoothing_spline/data',
    install_dir: py.get_install_dir() / 'smoothing_spline'
)
