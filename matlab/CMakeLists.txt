# Compile the MEX extensions.

# Based on FLANN code by Marius Muja:
#   https://github.com/mariusmuja/flann/blob/master/src/matlab/CMakeLists.txt

# Note that since Matlab ships Boost libraries with the same name as system
# libraries, but doesn't ship headers, we need to be careful not to cross-link.
#
# There are a few strategies for doing this:
#
#   - Get the headers for the appropriate version of Boost shipped by Matlab
#     and link all targets against Matlab's Boost.
#
#   - Statically link against the system Boost.
#
#   - Dynamically link against system Boost, and ensure that LD_LIBRARY_PATH or
#     maybe LD_PRELOAD or DYLD_LIBRARY_PATH (on Mac) is set to pick up system
#     Boost before it gets Matlab's Boost on runtime. This can cause Matlab
#     itself to crash.
#
# Static linking is the easiest approach, but unfortunately there doesn't seem
# to be a way to find both static and dynamic versions of boost, and not all
# systems have static versions of Boost with position-independent code. So we
# hack it by string-replacing .so/.dylib with .a in the path to the primary
# boost thread. Set the option MEX_BOOST_THREAD to the correct path if this
# doesn't work for you.
#
# If you want to dynamically link against Matlab boost, set BOOST_HEADERS to
# the location with the appropriate version of the boost headers and set
# BOOST_THREAD and BOOST_PROGRAM_OPTIONS to point to Matlab's version of those
# libraries, e.g. /path/to/matlab/bin/glnxa64/libboost_thread.so.1.44.0, as
# well as setting MEX_DYNAMIC_BOOST to ON.
#
# All targets will then be linked against Matlab's version of Boost; you'll
# probably have to be careful with linking other things against them. On OS X,
# where Matlab doesn't version-number its distribution of the boost libraries
# for some reason, you'll have to be especially careful with it attempting
# to load the system boost, which will probably result in segfaults or
# hanging-forever behavior.

# TODO: figure out what options there are for np-divs linking to boosts

# TODO: better approach to find static boost

# TODO: support octave

option(MEX_DYNAMIC_BOOST "Use the same boost for MEX as other targets" OFF)

string(REGEX REPLACE "(\\.dylib|\\.so)$" ".a"
    BOOST_THREAD_STATIC_GUESS "${BOOST_THREAD}")

set(MEX_BOOST_THREAD CACHE PATH
    "Path to Boost Thread to use for the MEX binary")

if(MEX_BOOST_THREAD)
    set(MEX_BOOST_THREAD_REAL ${MEX_BOOST_THREAD})
elseif(MEX_DYNAMIC_BOOST)
    set(MEX_BOOST_THREAD_REAL ${BOOST_THREAD})
else()
    set(MEX_BOOST_THREAD_REAL ${BOOST_THREAD_STATIC_GUESS})
endif()


set(MEX_NAME sdm_mex)

if(WIN32)
    find_program(MEX_CMD mex.bat)
    find_program(MEXEXT_CMD mexext.bat)
else()
    find_program(MEX_CMD mex)
    find_program(MEXEXT_CMD mexext)
endif()

if (MEX_CMD AND MEXEXT_CMD)
    get_filename_component(MEX_REAL_CMD ${MEX_CMD} ABSOLUTE)
    get_filename_component(MEX_PATH ${MEX_REAL_CMD} PATH)

    get_filename_component(MEXEXT_REAL_CMD ${MEXEXT_CMD} ABSOLUTE)
    get_filename_component(MEXEXT_PATH ${MEXEXT_REAL_CMD} PATH)

    if (MEX_PATH STREQUAL MEXEXT_PATH)
        message(STATUS "Found MATLAB at: " ${MEX_PATH})

        execute_process(COMMAND ${MEXEXT_REAL_CMD}
            OUTPUT_VARIABLE MEX_EXTENSION OUTPUT_STRIP_TRAILING_WHITESPACE) 
        set(MEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${MEX_NAME}.${MEX_EXTENSION})

        message("Mex: linking against ${MEX_BOOST_THREAD_REAL}")
        message("Boost thread: ${BOOST_THREAD}")
        message("Boost program options: ${BOOST_PROGRAM_OPTIONS}")
        message("Boost headers: ${BOOST_HEADERS}")

        set(MEX_ARGS "${CMAKE_CURRENT_SOURCE_DIR}/${MEX_NAME}.cpp" "-g")

        # -I args
        foreach(i ${PROJECT_SOURCE_DIR} ${BOOST_HEADERS} ${NPDIVS_INCLUDE_DIRS})
            list(APPEND MEX_ARGS "-I${i}")
        endforeach(i)

        # -L args
        foreach(i ${NPDIVS_LIBRARY_DIRS}
                  ${FLANN_LIBRARY_DIRS})
            list(APPEND MEX_ARGS "-L${i}")
        endforeach(i)

        # -l args
        foreach(i ${NPDIVS_LIBRARIES} ${FLANN_LIBRARIES})
            list(APPEND MEX_ARGS "-l${i}")
        endforeach(i)

        # direct arguments
        get_property(sdm_s_location TARGET sdm_s PROPERTY LOCATION)
        list(APPEND MEX_ARGS
            ${sdm_s_location}
            ${MEX_BOOST_THREAD_REAL}
            ${LIBSVM_LIBRARIES}
            ${CMAKE_THREAD_LIBS_INIT})

        # lapack/blas needs trickery to make mex happy on mac
        if(${APPLE})
            list(APPEND MEX_ARGS
                \"CXXLIBS=\\$$CXXLIBS -framework Accelerate\")
        else()
            foreach(i ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
                list(APPEND MEX_ARGS "${i}")
            endforeach(i)
        endif()

        add_custom_command(
            OUTPUT ${MEX_FILE}
            COMMAND ${MEX_REAL_CMD} ARGS ${MEX_ARGS}
            DEPENDS sdm_s ${CMAKE_CURRENT_SOURCE_DIR}/${MEX_NAME}.cpp
            COMMENT "Building MEX extension ${MEX_FILE}"
        )
        add_custom_target(sdm-mex ALL DEPENDS ${MEX_FILE})
        # TODO: export target?

        file(GLOB MATLAB_SOURCES *.m)
        install(FILES ${MEX_FILE} ${MATLAB_SOURCES}
            DESTINATION share/sdm/matlab
        )

    else()
        message(WARNING "The 'mex' and 'mexext' programs have been found in different locations; something is probably wrong. (Is 'mex' the pdftex alias?) Make sure MATLAB's 'bin' directory is in your PATH.\n${MEX_REAL_CMD}\n${MEXEXT_REAL_CMD}")
        set(BUILD_MATLAB_BINDINGS OFF)
    endif()

else()
    message(WARNING "Cannot find MATLAB installation. Make sure MATLAB's 'bin' directory is in your PATH.")
        set(BUILD_MATLAB_BINDINGS OFF)
endif()
