
# Different test cases may share some input files, for example grid file.
# Just keep one copy of these files.
SET(_shared_input_dir  ${CMAKE_CURRENT_SOURCE_DIR})
SET(_shared_input_file_list 2412.msh)

###
SET (_input_file_list input.prm)

###
SET (_output_file iter_history.out)
SET (_comparison_file iter_history.out.reference)

###
# SET (_exec_run ${CMAKE_BINARY_DIR}/bin/${PRJ_NAME})
SET (_para_run ${_input_file_list} > screen.log 2>&1)

###
GET_FILENAME_COMPONENT(_current_dir_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
SET(_test_name_prefix ${_current_dir_name})
UNSET(_current_dir_name)

PICKUP_ALL_SUBDIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

