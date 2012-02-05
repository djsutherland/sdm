cmake_minimum_required(VERSION 2.6)
project (sdm)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include_directories(${PROJECT_SOURCE_DIR})

# TODO: find np-divs

find_package(LibSVM REQUIRED)
include_directories(${LIBSVM_INCLUDE_DIRS})
link_directories(${LIBSVM_LIBRARY_DIRS})

find_package(Boost REQUIRED)
include_directories(${BOOST_INCLUDE_DIRS})
link_directories(${BOOST_LIBRARY_DIRS})

find_package(PkgConfig REQUIRED)
pkg_check_modules(FLANN REQUIRED flann)
include_directories(${FLANN_INCLUDE_DIRS})
link_directories(${FLANN_LIBRARY_DIRS})
if(FLANN_CFLAGS)
    add_definitions(${FLANN_CFLAGS})
endif(FLANN_CFLAGS)


enable_language(Fortran)

find_package(BLAS REQUIRED)
include_directories(${BLAS_INCLUDE_DIRS})
link_directories(${BLAS_LIBRARY_DIRS})

find_package(LAPACK REQUIRED)
include_directories(${LAPACK_INCLUDE_DIRS})
link_directories(${LAPACK_LIBRARY_DIRS})


add_subdirectory(sdm)
