# Builds the qoamax library and, as the top-level project, the CLI.
#
# Use it as a subproject (dependency('qoamax'), or the variable qoamax_dep), or after installing through pkg-config.
project(
  'qoamax',
  'c',
  version: '0.2.0', # Keep in sync with QOAMAX_VERSION_* in src/qoamax.h (checked by make test-build-systems).
  license: 'EUPL-1.2',
  license_files: 'LICENSE.md',
  meson_version: '>=1.4.0', # c_std=c23
  default_options: ['buildtype=release'], # The encoder is much slower without optimizations.
)

cc = meson.get_compiler('c')
if cc.get_id() == 'msvc'
  error('qoamax needs C23 and GCC vector extensions, which MSVC doesn\'t support. Use clang-cl instead.')
endif
qoamax_c_args = cc.get_supported_arguments('-Wno-unknown-attributes', '-Wno-unknown-pragmas')

# Only the library's source is compiled as C23. Its header works with C11 and C++11 or later.
qoamax_lib = library(
  'qoamax',
  'src/qoamax.c',
  c_args: qoamax_c_args,
  override_options: ['c_std=c23'],
  version: meson.project_version(),
  soversion: '0.2', # Before 1.0, a minor version may change the ABI.
  install: true,
)
install_headers('src/qoamax.h', 'src/qoamax_plain.h')

qoamax_dep = declare_dependency(link_with: qoamax_lib, include_directories: include_directories('src'))
meson.override_dependency('qoamax', qoamax_dep)

import('pkgconfig').generate(
  qoamax_lib,
  description: 'High-quality QOA (Quite OK Audio) encoder and decoder library',
  url: 'https://gitlab.com/BlobCodes/qoamax',
)

# The CLI includes the library's source (a single translation unit), it doesn't link the library.
if get_option('cli').enabled() or (get_option('cli').auto() and not meson.is_subproject())
  executable(
    'qoamax',
    'bin/qoamax.c',
    c_args: qoamax_c_args,
    override_options: ['c_std=c23'],
    dependencies: cc.find_library('m', required: false),
    install: true,
  )
endif
