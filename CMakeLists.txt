cmake_minimum_required(VERSION 3.0)
project(postllm)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()
set(PostgreSQL_ROOT "/usr/include/postgresql/16/server")
find_package(PostgreSQL REQUIRED)

if(NOT PostgreSQL_FOUND)
    message(FATAL_ERROR "PostgreSQL not found.")
endif()

# Add llama.cpp/common source files directly to the project since they are PRIVATE
set(COMMON_SOURCES
    llama.cpp/common/build-info.cpp
    llama.cpp/common/common.h
    llama.cpp/common/common.cpp
    llama.cpp/common/base64.hpp
    llama.cpp/common/sampling.h
    llama.cpp/common/sampling.cpp
    llama.cpp/common/console.h
    llama.cpp/common/console.cpp
    llama.cpp/common/grammar-parser.h
    llama.cpp/common/grammar-parser.cpp
    llama.cpp/common/json.hpp
    llama.cpp/common/json-schema-to-grammar.cpp
    llama.cpp/common/train.h
    llama.cpp/common/train.cpp
    llama.cpp/common/ngram-cache.h
    llama.cpp/common/ngram-cache.cpp
)

FILE(GLOB LIB_SOURCES "postllm.cpp")
add_library(postllm SHARED ${LIB_SOURCES} ${COMMON_SOURCES})
set_target_properties(postllm PROPERTIES PREFIX "")

find_program(PG_CONFIG_EXECUTABLE pg_config)
if(PG_CONFIG_EXECUTABLE)
    # set postgres share dir
    execute_process( COMMAND ${PG_CONFIG_EXECUTABLE} --sharedir OUTPUT_VARIABLE PostgreSQL_SHARE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE )
    # set postgres pkglib dir
    execute_process( COMMAND ${PG_CONFIG_EXECUTABLE} --pkglibdir OUTPUT_VARIABLE PostgreSQL_PKGLIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE )
else()
    message(ERROR "pg_config not exist")
endif()

include_directories(${PostgreSQL_INCLUDE_DIRS})
target_link_libraries(postllm PUBLIC ${PostgreSQL_LIBRARIES})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common)

find_library(LLAMA_LIB llama HINTS ${CMAKE_INSTALL_PREFIX}/lib)

set(TARGET postllm)
target_link_libraries(${TARGET} PRIVATE ${LLAMA_LIB})

install(FILES postllm.so
        DESTINATION "${PostgreSQL_PKGLIB_DIR}/"
)

install(FILES postllm.so
        DESTINATION "${PostgreSQL_PKGLIB_DIR}/postllm"
)

# install control file and sql file
install(FILES postllm.control postllm--1.0.sql
        DESTINATION "${PostgreSQL_SHARE_DIR}/extension"
)