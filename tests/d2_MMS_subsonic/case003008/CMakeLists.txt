SET (_test_directory ${CMAKE_CURRENT_BINARY_DIR})
SET (_test_src_directory ${CMAKE_CURRENT_SOURCE_DIR})


## The commented variables are put here just as a reminder.
## They are setted in the upper level CMakeList.txt and will
## be used inside macro SETUP_TEST_CASE below.
## You can uncomment some of them and over write their value
## if necessary.


# SET (_exec_copy ln )
# SET (_para_copy -s )

# SET (_exec_diff diff )
# SET (_para_diff "" )


# SET (_input_file_list input.prm.2412)

# SET(_shared_input_dir ${_shared_input_dir})
# SET(_shared_input_file_list ${_shared_input_file_list})

# SET (_output_file screen.log)
# SET (_comparison_file screen.log.reference)

SET (_exec_run  ${_exec_mpi})
SET (_para_run  -n 3 ${CMAKE_BINARY_DIR}/bin/${PRJ_NAME} ${_input_file_list} > screen.log 2>&1)

# _test_name_prefix is setted in the parent CMakeList.txt.
GET_FILENAME_COMPONENT(_current_dir_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
SET (_test_name ${_test_name_prefix}_${_current_dir_name})
UNSET(_current_dir_name)


# This macro seems take no parameter, but actually all variables appeared above
# are necessary for SETUP_TEST_CASE() to run properly.
SETUP_TEST_CASE()
