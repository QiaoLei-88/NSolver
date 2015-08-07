
SET (_lib_source_file_list "")
LIST(APPEND _lib_source_file_list assemble.cpp)
LIST(APPEND _lib_source_file_list compute.cpp)
LIST(APPEND _lib_source_file_list linearVelocityPotential.cpp)
LIST(APPEND _lib_source_file_list linearVelocityPotential_Postprocessor.cpp)
LIST(APPEND _lib_source_file_list output_result.cpp)
LIST(APPEND _lib_source_file_list setup_system.cpp)
LIST(APPEND _lib_source_file_list solve.cpp)
LIST(APPEND _lib_source_file_list transfer_solution.cpp)



GET_FILENAME_COMPONENT(__lib_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
ADD_LIBRARY (${__lib_name} STATIC ${_lib_source_file_list})
DEAL_II_SETUP_TARGET(${__lib_name})
