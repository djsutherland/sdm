cmake_minimum_required(VERSION 2.8)
project (SDM)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(EXE_LINK_SHARED
    "Link the executable to the shared library, rather than the static." ON)
option(BUILD_MATLAB_BINDINGS "Build the MEX interface for use with MATLAB." ON)

# version info
set(SDM_MAJOR_VERSION 0)
set(SDM_MINOR_VERSION 1)
set(SDM_PATCH_VERSION 0)
set(SDM_VERSION
    ${SDM_MAJOR_VERSION}.${SDM_MINOR_VERSION}.${SDM_PATCH_VERSION})

# offer the user the choice of overriding the installation directories
set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR include CACHE PATH
    "Installation directory for header files")
if(WIN32 AND NOT CYGWIN)
    set(DEF_INSTALL_CMAKE_DIR CMake)
else()
    set(DEF_INSTALL_CMAKE_DIR lib/CMake/SDM)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
    "Installation directory for CMake files")

# make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE CMAKE)
    set(var INSTALL_${p}_DIR)
    if(NOT IS_ABSOLUTE "${${var}}")
        set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
    endif()
endforeach()

# find dependencies, etc

include_directories(BEFORE ${PROJECT_SOURCE_DIR})

find_package(NPDivs REQUIRED)
include_directories(${NPDIVS_INCLUDE_DIRS})

find_package(LibSVM REQUIRED)
include_directories(${LIBSVM_INCLUDE_DIRS})

find_package(Threads)

find_package(Boost COMPONENTS thread program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(PkgConfig REQUIRED)
pkg_check_modules(FLANN REQUIRED flann)
include_directories(${FLANN_INCLUDE_DIRS})
if(FLANN_CFLAGS)
    add_definitions(${FLANN_CFLAGS})
endif(FLANN_CFLAGS)


enable_language(Fortran)

find_package(BLAS REQUIRED)
include_directories(${BLAS_INCLUDE_DIRS})

find_package(LAPACK REQUIRED)
include_directories(${LAPACK_INCLUDE_DIRS})

# subdirectories with actual content
add_subdirectory(sdm)
add_subdirectory(test)
if(BUILD_MATLAB_BINDINGS)
    add_subdirectory(matlab)
endif()


################################################################################
## stuff for making SDMConfig.cmake

################################################################################
## stuff for making NPDivsConfig.cmake

# add targets to build-tree export set
export(TARGETS sdm sdm_s sdm-run
    FILE "${PROJECT_BINARY_DIR}/SDMLibraryDepends.cmake")

# register package with cmake registry
export(PACKAGE SDM)

# create SDM{BuildTreeSettings,Config,ConfigVersion}.cmake
configure_file(SDMBuildTreeSettings.cmake.in
    "${PROJECT_BINARY_DIR}/SDMBuildTreeSettings.cmake" @ONLY)

file(RELATIVE_PATH CONF_REL_INCLUDE_DIR
    "${INSTALL_CMAKE_DIR}" "${INSTALL_INCLUDE_DIR}")
file(RELATIVE_PATH CONF_REL_LIB_DIR
    "${INSTALL_CMAKE_DIR}" "${INSTALL_LIB_DIR}")
configure_file(SDMConfig.cmake.in
    "${PROJECT_BINARY_DIR}/SDMConfig.cmake" @ONLY)
configure_file(SDMConfigVersion.cmake.in
    "${PROJECT_BINARY_DIR}/SDMConfigVersion.cmake" @ONLY)

# install export set for use with the install-tree
install(EXPORT SDMLibraryDepends
    DESTINATION "${INSTALL_CMAKE_DIR}"
    COMPONENT dev
)
install(
    FILES "${CMAKE_BINARY_DIR}/SDMConfig.cmake"
          "${CMAKE_BINARY_DIR}/SDMConfigVersion.cmake"
    DESTINATION "${INSTALL_CMAKE_DIR}"
    COMPONENT dev
)
