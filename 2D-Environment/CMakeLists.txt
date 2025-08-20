cmake_minimum_required(VERSION 3.20)
project(PacmanGame LANGUAGES CXX)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -------- Fetch raylib --------
include(FetchContent)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_GAMES    OFF CACHE BOOL "" FORCE)
FetchContent_Declare(raylib
        GIT_REPOSITORY https://github.com/raysan5/raylib.git
        GIT_TAG        5.0
        GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(raylib)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_SOURCE_DIR}/app/*.cpp"
        "${CMAKE_SOURCE_DIR}/core/*.cpp"
        "${CMAKE_SOURCE_DIR}/entities/*.cpp"
        "${CMAKE_SOURCE_DIR}/render/*.cpp"
)

message(STATUS "SOURCES collected: ${SOURCES}")

add_executable(PacmanGame ${SOURCES})

target_include_directories(PacmanGame PRIVATE
        "${CMAKE_SOURCE_DIR}/src"
        "${CMAKE_SOURCE_DIR}/src/app"
        "${CMAKE_SOURCE_DIR}/src/core"
        "${CMAKE_SOURCE_DIR}/src/entities"
        "${CMAKE_SOURCE_DIR}/src/render"
        "${CMAKE_SOURCE_DIR}/src/third_party"
)

target_link_libraries(PacmanGame PRIVATE raylib)

if (WIN32)
    target_compile_definitions(PacmanGame PRIVATE _WIN32_WINNT=0x0A00)
    target_link_libraries(PacmanGame PRIVATE ws2_32)
endif()

if (APPLE)
    target_link_libraries(PacmanGame PRIVATE
            "-framework OpenGL" "-framework Cocoa" "-framework IOKit" "-framework CoreVideo"
    )
endif()

add_custom_command(TARGET PacmanGame POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/resources"
        "$<TARGET_FILE_DIR:PacmanGame>/resources"
)