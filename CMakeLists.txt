cmake_minimum_required(VERSION 3.15)
project(SampleRateRNN)
set(CMAKE_CXX_STANDARD 17)

add_executable(sample_rate_rnn
    main.cpp
    ResampledRNN.cpp
)

# Fectch matplotlib-cpp
message(STATUS "Fetching matplotlib-cpp")
file(DOWNLOAD https://raw.githubusercontent.com/lava/matplotlib-cpp/master/matplotlibcpp.h ${CMAKE_CURRENT_BINARY_DIR}/matplotlib-cpp/matplotlibcpp.h)
find_package(Python3 COMPONENTS Development)

# add RTNeural
set(RTNEURAL_XSIMD ON CACHE BOOL "Use RTNeural with XSIMD backend" FORCE)
add_subdirectory(../RTNeural rtneural-build EXCLUDE_FROM_ALL)


target_include_directories(sample_rate_rnn
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/matplotlib-cpp
        ${Python3_INCLUDE_DIRS}
)

target_compile_definitions(sample_rate_rnn
    PRIVATE
        WITHOUT_NUMPY=1
)

target_link_libraries(sample_rate_rnn
    PRIVATE
        Python3::Python
        RTNeural
)
