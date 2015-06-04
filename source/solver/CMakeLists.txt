
SET (_lib_source_file_list "")
LIST(APPEND _lib_source_file_list NSolver.cpp)
LIST(APPEND _lib_source_file_list integrate_force.cpp)


GET_FILENAME_COMPONENT(__lib_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
ADD_LIBRARY (${__lib_name} STATIC ${_lib_source_file_list})
DEAL_II_SETUP_TARGET(${__lib_name})
