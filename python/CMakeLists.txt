configure_file(setup.py.tpl setup.py)

install(DIRECTORY sdm DESTINATION share/sdm/python)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/setup.py DESTINATION share/sdm/python)

find_package(PythonInterp)

if (PYTHON_EXECUTABLE)
    install(CODE "execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/setup.py install
        WORKING_DIRECTORY \"${CMAKE_CURRENT_SOURCE_DIR}\")")
endif()
