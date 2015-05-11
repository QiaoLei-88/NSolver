MACRO(SETUP_TEST_CASE)
# The execution flow of this macro is driven by dependency relationship 
# and runs from bottom to top.

  ENABLE_TESTING()

  # Prepair both shared and exclusive input files for running test case.
  # Also accumulate denpence file list.
  SET (_input_file_list_full "")

  FOREACH (__file ${_shared_input_file_list})
    ADD_CUSTOM_COMMAND(OUTPUT  ${_test_directory}/${__file}
      COMMAND ${_exec_copy}  ${_para_copy} ${_shared_input_dir}/${__file} ${_test_directory}/${__file}
      WORKING_DIRECTORY
        ${_test_directory}
      VERBATIM
      )
    LIST(APPEND _input_file_list_full ${_test_directory}/${__file})
  ENDFOREACH(__file)
  UNSET(__file)

  FOREACH (__file ${_input_file_list})
    ADD_CUSTOM_COMMAND(OUTPUT  ${_test_directory}/${__file}
      COMMAND ${_exec_copy}  ${_para_copy} ${_test_src_directory}/${__file} ${_test_directory}/${__file}
      WORKING_DIRECTORY
        ${_test_directory}
      VERBATIM
      )
    LIST(APPEND _input_file_list_full ${_test_directory}/${__file})
  ENDFOREACH(__file)
  UNSET(__file)

  # Run test case
  ADD_CUSTOM_COMMAND(OUTPUT ${_test_directory}/${_output_file}
    COMMAND  ${_exec_run} ${_para_run}
    WORKING_DIRECTORY
      ${_test_directory}
    DEPENDS
      ${_input_file_list_full}
      # Remove dependency on the project main target to avoid race condition
      # in parallel testing.
      # Always build the excutable before run cTest.
      # ${PRJ_NAME} 
    VERBATIM
    )

  # Bring ${_comparison_file} into ${_test_directory} for possible manuel comparision.
    ADD_CUSTOM_COMMAND(OUTPUT ${_test_directory}/${_comparison_file}
    COMMAND  ${_exec_copy}  ${_para_copy} ${_test_src_directory}/${_comparison_file} ${_test_directory}/${_comparison_file}
    WORKING_DIRECTORY
      ${_test_directory}
    DEPENDS
      ${_test_src_directory}/${_comparison_file}
    VERBATIM
    )
  # Compare output
  ADD_CUSTOM_TARGET(diff_${_test_name}
    COMMAND ${_exec_diff} ${_para_diff} ${_test_directory}/${_output_file}  ${_test_directory}/${_comparison_file} > output.diff 2>&1
    WORKING_DIRECTORY
      ${_test_directory}
    DEPENDS
      ${_test_directory}/${_output_file}
      ${_test_directory}/${_comparison_file}
    )

  # Finally add test target. This will be invoked by the ctest command.
  ADD_TEST(NAME ${_test_name}
    # The next command is equavalent to run 'make diff_${_test_name}' inside
    # ${CMAKE_BINARY_DIR}.
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target diff_${_test_name}
    WORKING_DIRECTORY ${_test_directory}
    )

  SET_TESTS_PROPERTIES(${_test_name}
    PROPERTIES TIMEOUT ${_time_out_time}
    )

ENDMACRO()
