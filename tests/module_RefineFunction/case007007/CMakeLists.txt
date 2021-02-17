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

SET (_exec_run ${_exec_mpi})
SET (_n_slots  4)
# SET (_para_run input.prm 2>&1 > screen.log || true)


SETUP_MODULE_TEST_CASE()
