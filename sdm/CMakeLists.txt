set(LIBRARY_SOURCES
    sdm_model.cpp
    sdm_c.cpp
    crossvalidate.cpp
    tune_params.cpp
    kernel_projection.cpp
    kernels/kernel.cpp
    kernels/linear.cpp
    kernels/polynomial.cpp
    kernels/gaussian.cpp
    kernels/from_str.cpp
)

set(LIBS
    ${NPDIVS_LIBRARIES}
    ${LIBSVM_LIBRARIES}
    ${BOOST_SYSTEM}
    ${BOOST_THREAD}
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
    target_link_libraries(sdm-run sdm ${BOOST_PROGRAM_OPTIONS})
else()
    target_link_libraries(sdm-run sdm_s
        ${LIBS} ${CMAKE_THREAD_LIBS_INIT} ${BOOST_PROGRAM_OPTIONS})
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
    FILES_MATCHING REGEX "/.*\\.h(pp)?$"
)
