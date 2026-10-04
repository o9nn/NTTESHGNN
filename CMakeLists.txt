cmake_minimum_required(VERSION 3.16)
project(NTTESHGNN VERSION 0.1.0 LANGUAGES C)

# ============================================================================
# OPTIONS
# ============================================================================

option(NT_BUILD_TESTS "Build test suite" ON)
option(NT_BUILD_EXAMPLES "Build examples" ON)
option(NT_BUILD_FIBER_GGML_DEMO "Build opt-in real GGML QKV fiber experiment" OFF)
set(GGML_SOURCE_DIR "" CACHE PATH "Pinned GGML source tree for opt-in fiber experiment")
option(NT_ENABLE_CUDA "Enable CUDA backend" OFF)
option(NT_ENABLE_METAL "Enable Metal backend" OFF)
option(NT_ENABLE_VTNPU "Enable VTNPU backend" OFF)
option(NT_ENABLE_SIMD "Enable SIMD optimizations" ON)
option(NT_ENABLE_OPENMP "Enable OpenMP parallelization" OFF)

# ============================================================================
# COMPILER SETTINGS
# ============================================================================

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(NT_ENABLE_SIMD)
        add_compile_options(-march=native)
    endif()
endif()

if(MSVC)
    add_compile_options(/W4)
endif()

# ============================================================================
# LIBRARY
# ============================================================================

set(NTTESHGNN_SOURCES
    src/nt_types.c
    src/nt_storage.c
    src/nt_tensor.c
    src/nt_context.c
    src/nt_ops.c
    src/nt_hypergraph.c
    src/nt_echostate.c
)

set(NTTESHGNN_HEADERS
    include/ntteshgnn/ntteshgnn.h
    include/ntteshgnn/nt_types.h
    include/ntteshgnn/nt_storage.h
    include/ntteshgnn/nt_tensor.h
    include/ntteshgnn/nt_nested.h
    include/ntteshgnn/nt_typesystem.h
    include/ntteshgnn/nt_hypergraph.h
    include/ntteshgnn/nt_echostate.h
    include/ntteshgnn/nt_context.h
    include/ntteshgnn/nt_ops.h
    include/ntteshgnn/nt_vtnpu.h
)

# Create library
add_library(ntteshgnn ${NTTESHGNN_SOURCES})

target_include_directories(ntteshgnn
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link math library on Unix
if(UNIX)
    target_link_libraries(ntteshgnn PUBLIC m)
endif()

# OpenMP
if(NT_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_C_FOUND)
        target_link_libraries(ntteshgnn PUBLIC OpenMP::OpenMP_C)
        target_compile_definitions(ntteshgnn PUBLIC NT_ENABLE_OPENMP)
    endif()
endif()

# CUDA
if(NT_ENABLE_CUDA)
    enable_language(CUDA)
    target_compile_definitions(ntteshgnn PUBLIC NT_ENABLE_CUDA)
endif()

# Metal
if(NT_ENABLE_METAL AND APPLE)
    target_compile_definitions(ntteshgnn PUBLIC NT_ENABLE_METAL)
    target_link_libraries(ntteshgnn PUBLIC "-framework Metal" "-framework Foundation")
endif()

# VTNPU
if(NT_ENABLE_VTNPU)
    target_compile_definitions(ntteshgnn PUBLIC NT_ENABLE_VTNPU)
endif()

# Set library version
set_target_properties(ntteshgnn PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# ============================================================================
# TESTS
# ============================================================================

if(NT_BUILD_TESTS)
    enable_testing()
    
    add_executable(test_tensor tests/test_tensor.c)
    target_link_libraries(test_tensor PRIVATE ntteshgnn)
    add_test(NAME test_tensor COMMAND test_tensor)

    add_executable(test_layout tests/test_layout.c)
    target_link_libraries(test_layout PRIVATE ntteshgnn)
    add_test(NAME test_layout COMMAND test_layout)
    
    add_executable(test_ops tests/test_ops.c)
    target_link_libraries(test_ops PRIVATE ntteshgnn)
    add_test(NAME test_ops COMMAND test_ops)
    
    add_executable(test_hypergraph tests/test_hypergraph.c)
    target_link_libraries(test_hypergraph PRIVATE ntteshgnn)
    add_test(NAME test_hypergraph COMMAND test_hypergraph)
    
    add_executable(test_esn tests/test_esn.c)
    target_link_libraries(test_esn PRIVATE ntteshgnn)
    add_test(NAME test_esn COMMAND test_esn)
endif()

# ============================================================================
# EXAMPLES
# ============================================================================

if(NT_BUILD_EXAMPLES)
    add_executable(example_basic examples/basic.c)
    target_link_libraries(example_basic PRIVATE ntteshgnn)
    
    add_executable(example_gnn examples/gnn.c)
    target_link_libraries(example_gnn PRIVATE ntteshgnn)
    
    add_executable(example_esn examples/esn.c)
    target_link_libraries(example_esn PRIVATE ntteshgnn)
endif()

if(NT_BUILD_FIBER_GGML_DEMO)
    if(NOT EXISTS "${GGML_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "Set GGML_SOURCE_DIR to a pinned GGML source tree")
    endif()
    enable_language(CXX)
    add_subdirectory("${GGML_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/external-ggml" EXCLUDE_FROM_ALL)
    add_executable(fiber_qkv_graph
        examples/federated_fiber/qkv_graph.cpp
        examples/federated_fiber/nt_bridge.c)
    target_compile_features(fiber_qkv_graph PRIVATE cxx_std_17)
    target_link_libraries(fiber_qkv_graph PRIVATE ntteshgnn ggml ggml-cpu)
    if(NT_BUILD_TESTS)
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        add_test(NAME test_fiber_qkv_graph
            COMMAND "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/examples/federated_fiber/test_graph.py"
                    "$<TARGET_FILE:fiber_qkv_graph>")
    endif()
endif()

# ============================================================================
# INSTALLATION
# ============================================================================

include(GNUInstallDirs)

install(TARGETS ntteshgnn
    EXPORT ntteshgnn-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ntteshgnn
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT ntteshgnn-targets
    FILE ntteshgnn-config.cmake
    NAMESPACE ntteshgnn::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ntteshgnn
)

# ============================================================================
# PACKAGE CONFIG
# ============================================================================

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ntteshgnn-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ntteshgnn-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ntteshgnn
)

# ============================================================================
# SUMMARY
# ============================================================================

message(STATUS "")
message(STATUS "NTTESHGNN Configuration Summary")
message(STATUS "================================")
message(STATUS "Version:      ${PROJECT_VERSION}")
message(STATUS "Build type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "C Compiler:   ${CMAKE_C_COMPILER_ID}")
message(STATUS "")
message(STATUS "Options:")
message(STATUS "  Tests:      ${NT_BUILD_TESTS}")
message(STATUS "  Examples:   ${NT_BUILD_EXAMPLES}")
message(STATUS "  CUDA:       ${NT_ENABLE_CUDA}")
message(STATUS "  Metal:      ${NT_ENABLE_METAL}")
message(STATUS "  VTNPU:      ${NT_ENABLE_VTNPU}")
message(STATUS "  SIMD:       ${NT_ENABLE_SIMD}")
message(STATUS "  OpenMP:     ${NT_ENABLE_OPENMP}")
message(STATUS "")
