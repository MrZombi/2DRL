cmake_minimum_required(VERSION 4.0)
project(2D-Environment LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_SOURCE_DIR}/app/*.cpp"
        "${CMAKE_SOURCE_DIR}/core/*.cpp"
        "${CMAKE_SOURCE_DIR}/entities/*.cpp"
)

message(STATUS "SOURCES collected: ${SOURCES}")

add_executable(2D-Environment ${SOURCES})

target_include_directories(2D-Environment PRIVATE
        "${CMAKE_SOURCE_DIR}/src"
        "${CMAKE_SOURCE_DIR}/src/app"
        "${CMAKE_SOURCE_DIR}/src/core"
        "${CMAKE_SOURCE_DIR}/src/entities"
        "${CMAKE_SOURCE_DIR}/src/third_party"
)

if (WIN32)
    target_compile_definitions(2D-Environment PRIVATE _WIN32_WINNT=0x0A00)
    target_link_libraries(2D-Environment PRIVATE ws2_32)
endif()