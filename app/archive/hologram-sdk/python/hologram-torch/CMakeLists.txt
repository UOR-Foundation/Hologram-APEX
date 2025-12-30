cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(hologram_torch CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find PyTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Find Python
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Hologram FFI library path (updated for hologram-sdk/python/hologram-torch location)
set(HOLOGRAM_ROOT "${CMAKE_SOURCE_DIR}/../../..")
set(HOLOGRAM_FFI_LIB "${HOLOGRAM_ROOT}/target/release/libhologram_ffi${CMAKE_SHARED_LIBRARY_SUFFIX}")

# Check if hologram-ffi library exists
if(NOT EXISTS ${HOLOGRAM_FFI_LIB})
    message(FATAL_ERROR "hologram-ffi library not found at ${HOLOGRAM_FFI_LIB}. "
                        "Please run 'cargo build --release --lib -p hologram-ffi' first.")
endif()

# Import hologram-ffi library
# Use IMPORTED_SONAME to specify just the library name for the NEEDED entry
add_library(hologram_ffi SHARED IMPORTED)
set_target_properties(hologram_ffi PROPERTIES
    IMPORTED_LOCATION ${HOLOGRAM_FFI_LIB}
    IMPORTED_NO_SONAME TRUE
    INTERFACE_INCLUDE_DIRECTORIES "${HOLOGRAM_ROOT}/crates/hologram-ffi/src"
)

# Add library directory to link path
link_directories("${HOLOGRAM_ROOT}/target/release")

# Hologram PyTorch extension sources
set(HOLOGRAM_TORCH_SOURCES
    csrc/hologram_backend.cpp
    csrc/hologram_tensor.cpp
    csrc/hologram_ops.cpp
    csrc/hologram_autograd.cpp
)

# Create shared library
add_library(hologram_torch_lib SHARED ${HOLOGRAM_TORCH_SOURCES})

# Include directories
target_include_directories(hologram_torch_lib PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/csrc
    ${HOLOGRAM_ROOT}/crates/hologram-ffi/src
)

# Link libraries
target_link_libraries(hologram_torch_lib
    ${TORCH_LIBRARIES}
    -lhologram_ffi
)

# Python extension module
set_target_properties(hologram_torch_lib PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_hologram_torch"
    SUFFIX "${Python_EXTENSION_SUFFIX}"
)

# Set RPATH for finding hologram_ffi at runtime
set(HOLOGRAM_FFI_DIR "${HOLOGRAM_ROOT}/target/release")
if(APPLE)
    set_target_properties(hologram_torch_lib PROPERTIES
        INSTALL_RPATH "@loader_path/../../../../..:@loader_path/../../../../../target/release"
        BUILD_WITH_INSTALL_RPATH TRUE
        BUILD_RPATH "${HOLOGRAM_FFI_DIR}"
    )
elseif(UNIX)
    set_target_properties(hologram_torch_lib PROPERTIES
        INSTALL_RPATH "$ORIGIN/../../../../..:$ORIGIN/../../../../../target/release"
        BUILD_WITH_INSTALL_RPATH TRUE
        BUILD_RPATH "${HOLOGRAM_FFI_DIR}"
    )
endif()

# Installation
install(TARGETS hologram_torch_lib
    LIBRARY DESTINATION hologram_torch
)
