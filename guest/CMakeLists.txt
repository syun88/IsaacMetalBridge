cmake_minimum_required(VERSION 3.25)
project(IsaacMetalBridgeGuest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(imb-guest-probe src/main.cpp)
target_include_directories(imb-guest-probe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../protocol/include
)
target_compile_options(imb-guest-probe PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_executable(imb-protocol-tests tests/protocol_tests.cpp)
target_include_directories(imb-protocol-tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../protocol/include
)
target_compile_options(imb-protocol-tests PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME imb-protocol-tests COMMAND imb-protocol-tests)
