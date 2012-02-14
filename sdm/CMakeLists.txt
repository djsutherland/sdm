add_library(sdm
    sdm.cpp
    kernel_projection.cpp
    kernels/kernel.cpp
    kernels/linear.cpp
    kernels/polynomial.cpp
    kernels/gaussian.cpp
)
target_link_libraries(sdm
    ${NPDIVS_LIBRARIES}
    ${LIBSVM_LIBRARIES}
    ${Boost_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)


add_executable(sdm-run
    cli.cpp
)
target_link_libraries(sdm-run sdm ${Boost_LIBRARIES})


install(
    TARGETS sdm sdm-run
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
