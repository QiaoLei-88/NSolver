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
  STRING(REPLACE ";" " " _para_run_str "${_para_run}")
  ADD_CUSTOM_COMMAND(OUTPUT ${_test_directory}/${_output_file}
    COMMAND ${CMAKE_SOURCE_DIR}/utilities/script/regTestDriver.sh ${_exec_run} "${_para_run_str}" ${_comparison_file} ${_output_file}
    WORKING_DIRECTORY
      ${_test_directory}
    DEPENDS
      ${_input_file_list_full}
      # Remove dependency on the project main target to avoid race condition
      # in parallel testing.
      # Always build the executable before run cTest.
      # ${PRJ_NAME}
      ${_extra_depends_for_output}
      ${_test_directory}/${_comparison_file}
    VERBATIM
    )
  UNSET(_para_run_str)

  # Bring ${_comparison_file} into ${_test_directory} for possible manual comparison.
  ADD_CUSTOM_COMMAND(OUTPUT ${_test_directory}/${_comparison_file}
    COMMAND  ${_exec_copy}  ${_para_copy} ${_test_src_directory}/${_comparison_file} ${_test_directory}/${_comparison_file}
    WORKING_DIRECTORY
      ${_test_directory}
    DEPENDS
      ${_test_src_directory}/${_comparison_file}
    VERBATIM
    )

  # Compare output
  STRING(REPLACE ";" " " _para_diff_str "${_para_diff}")
  ADD_CUSTOM_TARGET(diff_${_test_name}
    COMMAND ${CMAKE_SOURCE_DIR}/utilities/script/compareOutput.sh ${_exec_diff} "${_para_diff_str}" ${_comparison_file} ${_output_file}
    WORKING_DIRECTORY
      ${_test_directory}
    DEPENDS
      ${_test_directory}/${_output_file}
      ${_test_directory}/${_comparison_file}
    )
  UNSET(_para_diff_str)

  # Finally add test target. This will be invoked by the ctest command.
  ADD_TEST(NAME ${_test_name}
    # The next command is equivalent to run 'make diff_${_test_name}' inside
    # ${CMAKE_BINARY_DIR}.
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target diff_${_test_name}
    WORKING_DIRECTORY ${_test_directory}
    )

  SET_TESTS_PROPERTIES(${_test_name}
    PROPERTIES TIMEOUT ${_time_out_time}
    )

  # Code for cleaning
  SET_DIRECTORY_PROPERTIES(
    PROPERTIES
    ADDITIONAL_MAKE_CLEAN_FILES "${_additional_clean_up_files}"
    )

  ADD_CUSTOM_TARGET(clean_${_test_name}
    COMMAND ${CMAKE_COMMAND} -E remove -f ${_additional_clean_up_files}
    COMMAND ${CMAKE_COMMAND} -E remove -f output.diff
    COMMAND ${CMAKE_COMMAND} -E remove -f ${_extra_depends_for_output}
    COMMAND ${CMAKE_COMMAND} -E remove -f ${_output_file}
    COMMAND ${CMAKE_COMMAND} -E remove -f ${_input_file_list}
    COMMAND ${CMAKE_COMMAND} -E remove -f ${_shared_input_file_list}
    COMMAND ${CMAKE_COMMAND} -E remove -f ${_comparison_file}
    WORKING_DIRECTORY ${_test_directory}
    )

  ADD_DEPENDENCIES(clean_${_test_name_prefix}
    clean_${_test_name}
    )

ENDMACRO()
