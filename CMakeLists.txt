# Builds the qoamax library (target qoamax::qoamax) and, as the top-level project, the CLI.
#
# Use it with add_subdirectory(), FetchContent, or after installing with find_package(qoamax).
# Options: QOAMAX_BUILD_CLI, QOAMAX_INSTALL, and CMake's BUILD_SHARED_LIBS.
cmake_minimum_required(VERSION 3.21) # C_STANDARD 23

# The version is defined in the header.
foreach(part MAJOR MINOR PATCH)
  file(STRINGS src/qoamax.h line REGEX "^#define QOAMAX_VERSION_${part} ")
  string(REGEX REPLACE "^#define QOAMAX_VERSION_${part} ([0-9]+).*" "\\1" qoamax_version_${part} "${line}")
endforeach()

project(qoamax
  VERSION ${qoamax_version_MAJOR}.${qoamax_version_MINOR}.${qoamax_version_PATCH}
  DESCRIPTION "High-quality QOA (Quite OK Audio) encoder and decoder library"
  HOMEPAGE_URL "https://gitlab.com/BlobCodes/qoamax"
  LANGUAGES C)

option(QOAMAX_BUILD_CLI "Build the qoamax command line tool" ${PROJECT_IS_TOP_LEVEL})
option(QOAMAX_INSTALL "Install the library (and the CLI, if built)" ${PROJECT_IS_TOP_LEVEL})

# The encoder is much slower without optimizations.
if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC AND NOT CMAKE_C_COMPILER_ID STREQUAL "Clang")
  message(FATAL_ERROR "qoamax needs C23 and GCC vector extensions, which MSVC doesn't support. "
                      "Use clang-cl instead (e.g. cmake -T ClangCL).")
endif()

include(GNUInstallDirs)

# Only the library's source is compiled as C23. Its header works with C11 and C++11 or later.
add_library(qoamax src/qoamax.c)
add_library(qoamax::qoamax ALIAS qoamax)
target_include_directories(qoamax PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
                                         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_options(qoamax PRIVATE $<$<C_COMPILER_ID:Clang,AppleClang>:-Wno-unknown-attributes>
                                      $<$<C_COMPILER_ID:GNU,Clang,AppleClang>:-Wno-unknown-pragmas>)
set_target_properties(qoamax PROPERTIES
  C_STANDARD 23
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
  VERSION ${PROJECT_VERSION}
  # Before 1.0, a minor version may change the ABI.
  SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
  PUBLIC_HEADER "src/qoamax.h;src/qoamax_plain.h"
  WINDOWS_EXPORT_ALL_SYMBOLS ON)

if(QOAMAX_BUILD_CLI)
  # The CLI includes the library's source (a single translation unit), it doesn't link the library.
  add_executable(qoamax_cli bin/qoamax.c)
  target_compile_options(qoamax_cli PRIVATE $<$<C_COMPILER_ID:Clang,AppleClang>:-Wno-unknown-attributes>
                                            $<$<C_COMPILER_ID:GNU,Clang,AppleClang>:-Wno-unknown-pragmas>)
  set_target_properties(qoamax_cli PROPERTIES OUTPUT_NAME qoamax C_STANDARD 23 C_STANDARD_REQUIRED ON
                                              C_EXTENSIONS OFF)
  if(UNIX)
    target_link_libraries(qoamax_cli PRIVATE m)
  endif()
endif()

if(QOAMAX_INSTALL)
  install(TARGETS qoamax EXPORT qoamaxTargets)
  # The library has no dependencies, so the exported targets are the whole package config.
  install(EXPORT qoamaxTargets FILE qoamaxConfig.cmake NAMESPACE qoamax::
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/qoamax)
  include(CMakePackageConfigHelpers)
  write_basic_package_version_file(qoamaxConfigVersion.cmake COMPATIBILITY SameMinorVersion)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/qoamaxConfigVersion.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/qoamax)

  set(PREFIX ${CMAKE_INSTALL_PREFIX})
  set(LIBDIR ${CMAKE_INSTALL_FULL_LIBDIR})
  set(INCLUDEDIR ${CMAKE_INSTALL_FULL_INCLUDEDIR})
  set(VERSION ${PROJECT_VERSION})
  configure_file(qoamax.pc.in qoamax.pc @ONLY)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/qoamax.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

  if(QOAMAX_BUILD_CLI)
    install(TARGETS qoamax_cli)
  endif()
endif()
