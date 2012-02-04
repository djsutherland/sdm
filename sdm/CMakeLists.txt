add_library(sdm SHARED
    kernels.cpp
)
target_link_libraries(sdm ${LIBSVM_LIBRARIES} ${Boost_LIBRARIES})

install(
    TARGETS 
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(
    DIRECTORY ./
    DESTINATION include/sdm
    FILES_MATCHING PATTERN "*.hpp"
)
