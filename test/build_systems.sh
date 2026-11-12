#!/usr/bin/env bash
# Checks the build systems: builds and installs the library with the Makefile, CMake and Meson, and builds the
# README example (test/example.c, as C11) and the C++ test (test/test_cpp.cpp, as C++11), both with -pedantic-errors,
# against it in every supported way:
#   - Makefile: make install, consumers via pkg-config (shared and static)
#   - CMake:    install + find_package (shared), add_subdirectory (static)
#   - Meson:    install + pkg-config (shared), subproject (static)
# Usage: test/build_systems.sh WORK_DIR (deleted and recreated). Uses $CC and $CXX (default cc, c++).
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
work=$(realpath -m "${1:?usage: $0 WORK_DIR}")
export CC=${CC:-cc} CXX=${CXX:-c++}
rm -rf "$work"
mkdir -p "$work"

fail() {
  echo "build_systems.sh: $*" >&2
  exit 1
}
step() { echo "=== $*"; }

# Runs a build command, showing its output only if it fails.
quiet() {
  local log
  log=$("$@" 2>&1) || {
    echo "$log" >&2
    fail "failed: $*"
  }
}

# Runs the consumers in a directory. With a library directory, they must use the shared library from there.
run_consumers() {
  local dir=$1 shared_libdir=${2:-}
  for program in example test_cpp; do
    if [ -n "$shared_libdir" ]; then
      ldd "$dir/$program" | grep -q "$shared_libdir/libqoamax.so" || fail "$dir/$program doesn't use $shared_libdir"
    else
      ! ldd "$dir/$program" | grep -q libqoamax || fail "$dir/$program isn't linked statically"
    fi
    "$dir/$program" > /dev/null || fail "$dir/$program failed"
  done
}

# Builds the consumers with pkg-config from an installation prefix, linking the shared or static library.
pkg_config_consumers() {
  local prefix=$1 out=$2
  local pc_dir
  pc_dir=$(dirname "$(find "$prefix" -name qoamax.pc)")
  export PKG_CONFIG_PATH=$pc_dir
  local cflags libdir
  cflags=$(pkg-config --cflags qoamax)
  libdir=$(pkg-config --variable=libdir qoamax)
  pkg-config --exact-version="$version" qoamax || fail "pkg-config version isn't $version"

  mkdir -p "$out/shared"
  # shellcheck disable=SC2046,SC2086
  $CC -std=c11 -pedantic-errors -Wall -Wextra -Werror $cflags "$root/test/example.c" \
    $(pkg-config --libs qoamax) -Wl,-rpath,"$libdir" -o "$out/shared/example"
  # shellcheck disable=SC2046,SC2086
  $CXX -std=c++11 -pedantic-errors -Wall -Wextra -Werror $cflags "$root/test/test_cpp.cpp" \
    $(pkg-config --libs qoamax) -Wl,-rpath,"$libdir" -o "$out/shared/test_cpp"
  run_consumers "$out/shared" "$libdir"

  # Only the Makefile installs both libraries (CMake and Meson build one or the other).
  if [ -e "$libdir/libqoamax.a" ]; then
    mkdir -p "$out/static"
    # shellcheck disable=SC2086
    $CC -std=c11 -pedantic-errors -Wall -Wextra -Werror $cflags "$root/test/example.c" "$libdir/libqoamax.a" \
      -o "$out/static/example"
    # shellcheck disable=SC2086
    $CXX -std=c++11 -pedantic-errors -Wall -Wextra -Werror $cflags "$root/test/test_cpp.cpp" "$libdir/libqoamax.a" \
      -o "$out/static/test_cpp"
    run_consumers "$out/static"
  fi
  unset PKG_CONFIG_PATH
}

# The version is defined in the header. Meson can't read it from there, so check its copy.
version=$(awk '/^#define QOAMAX_VERSION_(MAJOR|MINOR|PATCH) / { printf "%s%s", sep, $3; sep = "." }' \
  "$root/src/qoamax.h")
meson_version=$(sed -n "s/^  version: '\([0-9.]*\)'.*/\1/p" "$root/meson.build")
[ "$version" = "$meson_version" ] || fail "meson.build has version $meson_version, src/qoamax.h has $version"
echo "qoamax $version"

step "Makefile: make install, consumers via pkg-config"
quiet make -C "$root" CC="$CC" BUILD_DIR="$work/make-build" PREFIX="$work/make-prefix" install
[ -x "$work/make-prefix/bin/qoamax" ] || fail "make install didn't install the CLI"
pkg_config_consumers "$work/make-prefix" "$work/make-consumer"

# A CMake project using the library through find_package, or add_subdirectory with QOAMAX_SOURCE_DIR.
mkdir -p "$work/cmake-consumer"
cat > "$work/cmake-consumer/CMakeLists.txt" << EOF
cmake_minimum_required(VERSION 3.21)
project(qoamax_consumer C CXX)
if(QOAMAX_SOURCE_DIR)
  add_subdirectory(\${QOAMAX_SOURCE_DIR} qoamax)
else()
  find_package(qoamax $version EXACT REQUIRED)
endif()
add_executable(example "$root/test/example.c")
set_target_properties(example PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
target_compile_options(example PRIVATE -pedantic-errors -Wall -Wextra -Werror)
target_link_libraries(example PRIVATE qoamax::qoamax)
add_executable(test_cpp "$root/test/test_cpp.cpp")
set_target_properties(test_cpp PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(test_cpp PRIVATE -pedantic-errors -Wall -Wextra -Werror)
target_link_libraries(test_cpp PRIVATE qoamax::qoamax)
EOF

step "CMake: install (shared), consumers via find_package and pkg-config"
quiet cmake -S "$root" -B "$work/cmake-build" -G Ninja -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_INSTALL_PREFIX="$work/cmake-prefix"
quiet cmake --build "$work/cmake-build"
quiet cmake --install "$work/cmake-build"
[ -x "$work/cmake-prefix/bin/qoamax" ] || fail "CMake didn't install the CLI"
quiet cmake -S "$work/cmake-consumer" -B "$work/cmake-find-package" -G Ninja \
  -DCMAKE_PREFIX_PATH="$work/cmake-prefix"
quiet cmake --build "$work/cmake-find-package"
cmake_libdir=$(dirname "$(find "$work/cmake-prefix" -name 'libqoamax.so' | head -n 1)")
run_consumers "$work/cmake-find-package" "$cmake_libdir"
pkg_config_consumers "$work/cmake-prefix" "$work/cmake-pkg-config"

step "CMake: add_subdirectory (static)"
quiet cmake -S "$work/cmake-consumer" -B "$work/cmake-subdirectory" -G Ninja -DQOAMAX_SOURCE_DIR="$root"
quiet cmake --build "$work/cmake-subdirectory"
[ ! -e "$work/cmake-subdirectory/qoamax/qoamax" ] || fail "CMake built the CLI as a subproject"
run_consumers "$work/cmake-subdirectory"

# A Meson project using the library through dependency(), installed or as a subproject.
mkdir -p "$work/meson-consumer/subprojects"
cp "$root/test/example.c" "$root/test/test_cpp.cpp" "$root/test/signal.h" "$work/meson-consumer/"
ln -s "$root" "$work/meson-consumer/subprojects/qoamax"
cat > "$work/meson-consumer/meson.build" << 'EOF'
project('qoamax_consumer', 'c', 'cpp',
        default_options: ['c_std=c11', 'cpp_std=c++11', 'warning_level=3', 'werror=true'])
qoamax_dep = dependency('qoamax', fallback: 'qoamax')
executable('example', 'example.c', dependencies: qoamax_dep, c_args: '-pedantic-errors')
executable('test_cpp', 'test_cpp.cpp', dependencies: qoamax_dep, cpp_args: '-pedantic-errors')
EOF

step "Meson: install (shared), consumers via dependency() and pkg-config"
quiet meson setup "$work/meson-build" "$root" --prefix="$work/meson-prefix"
quiet meson install -C "$work/meson-build"
[ -x "$work/meson-prefix/bin/qoamax" ] || fail "Meson didn't install the CLI"
meson_libdir=$(dirname "$(find "$work/meson-prefix" -name 'libqoamax.so' | head -n 1)")
PKG_CONFIG_PATH="$meson_libdir/pkgconfig" quiet meson setup "$work/meson-installed" "$work/meson-consumer" \
  -Dwrap_mode=nofallback
quiet meson compile -C "$work/meson-installed"
run_consumers "$work/meson-installed" "$meson_libdir"
pkg_config_consumers "$work/meson-prefix" "$work/meson-pkg-config"

step "Meson: subproject (static)"
quiet meson setup "$work/meson-subproject" "$work/meson-consumer" --force-fallback-for=qoamax \
  -Ddefault_library=static
quiet meson compile -C "$work/meson-subproject"
[ ! -e "$work/meson-subproject/subprojects/qoamax/qoamax" ] || fail "Meson built the CLI as a subproject"
run_consumers "$work/meson-subproject"

rm -rf "$work"
echo "build_systems.sh: all build systems work"
