cmake_minimum_required(VERSION 3.22)
project(mle_core VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_CUDA "Enable CUDA support" OFF)
option(BUILD_TESTS "Build tests" ON)

# Core library
add_library(mle_core STATIC
    src/loader.cpp
    src/engine.cpp
    src/ops_cpu.cpp
)

target_include_directories(mle_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(mle_core PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /O2>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -O3 -march=native>
)

# CUDA support
if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    target_sources(mle_core PRIVATE
        src/ops_cuda.cu
        src/cuda_context.cpp
    )
    
    target_link_libraries(mle_core PUBLIC
        CUDA::cudart
        CUDA::cublas
    )
    
    target_compile_definitions(mle_core PUBLIC ENABLE_CUDA)
    
    set_target_properties(mle_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "75;80;86;89"
    )
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    find_package(GTest)
    
    if(GTest_FOUND)
        add_executable(mle_tests
            tests/test_loader.cpp
            tests/test_ops.cpp
            tests/test_engine.cpp
        )
        
        target_link_libraries(mle_tests PRIVATE
            mle_core
            GTest::gtest
            GTest::gtest_main
        )
        
        add_test(NAME mle_tests COMMAND mle_tests)
    endif()
endif()

# Install
install(TARGETS mle_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include/mle
)
