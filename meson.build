project('jobsystem', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3', 'cpp_std=c++14'])

# These arguments are only used to build the shared library
# not the executables that use the library.
lib_args = ['-DBUILDING_JOBSYSTEM']

shlib = shared_library('jobsystem', ['jobsystem.cpp', 'Allocator.cpp'],
  install : true,
  cpp_args : lib_args,
  gnu_symbol_visibility : 'hidden',
  dependencies: [dependency('threads')]
)

test_exe = executable('jobsystem', 'jobsystem_test.cpp',
  link_with : shlib)
test('jobsystem', test_exe)

# Make this library usable as a Meson subproject.
jobsystem_dep = declare_dependency(
  include_directories: include_directories('.'),
  link_with : shlib)

# Make this library usable from the system's
# package manager.
install_headers('jobsystem.hpp', subdir : 'jobsystem')

pkg_mod = import('pkgconfig')
pkg_mod.generate(
  name : 'jobsystem',
  filebase : 'jobsystem',
  description : 'Meson sample project.',
  subdirs : 'jobsystem',
  libraries : shlib,
  version : '0.1',
)
