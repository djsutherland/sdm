add_library(sdm SHARED
    sdm.cpp
    kernels.cpp
    kernel_projection.cpp
)
target_link_libraries(sdm
    ${LIBSVM_LIBRARIES} ${Boost_LIBRARIES}
    ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
)

install(
    TARGETS sdm
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(
    DIRECTORY ./
    DESTINATION include/sdm
    FILES_MATCHING PATTERN "*.hpp"
)
