set(LIBRARY_SOURCES
    sdm.cpp
    kernel_projection.cpp
    kernels/kernel.cpp
    kernels/linear.cpp
    kernels/polynomial.cpp
    kernels/gaussian.cpp
)

set(LIBS
    ${NPDIVS_LIBRARIES}
    ${LIBSVM_LIBRARIES}
    ${Boost_THREAD_LIBRARY}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

add_library(sdm SHARED ${LIBRARY_SOURCES})
target_link_libraries(sdm ${LIBS})

add_library(sdm_s STATIC ${LIBRARY_SOURCES})
target_link_libraries(sdm_s ${LIBS} ${CMAKE_THREAD_LIBS_INIT})
if(CMAKE_COMPILER_IS_GNUCC)
    set_target_properties(sdm_s PROPERTIES COMPILE_FLAGS -fPIC)
endif()


add_executable(sdm-run cli.cpp)
if(EXE_LINK_SHARED)
    target_link_libraries(sdm-run sdm ${Boost_PROGRAM_OPTIONS_LIBRARY})
else()
    target_link_libraries(sdm-run sdm_s
        ${LIBS} ${CMAKE_THREAD_LIBS_INIT} ${Boost_PROGRAM_OPTIONS_LIBRARY})
endif()

install(
    TARGETS sdm sdm_s sdm-run
    EXPORT SDMLibraryDepends
    RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT shlib
    ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
    PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}/sdm" COMPONENT dev
)

install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DESTINATION "${INSTALL_INCLUDE_DIR}"
    COMPONENT dev
    FILES_MATCHING PATTERN "*.hpp"
)
