MACRO(SETUP_TEST_CATEGORY)
  SET(_shared_input_dir ${CMAKE_CURRENT_SOURCE_DIR})

  # _test_name_prefix is mandatory, i.e., can't be preset to other values.
  GET_FILENAME_COMPONENT(_current_dir_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
  SET(_test_name_prefix ${_current_dir_name})
  IF(NOT _current_dir_name MATCHES "^module_")
    LIST (APPEND _additional_clean_up_files ${_addtional_solver_outputs})
  ENDIF()

  # set default values to parameters
  IF(NOT DEFINED _src_file_list)
    IF(_current_dir_name MATCHES "^module_")
      SET (_src_file_list "main.cc")
    ELSE()
      SET (_src_file_list "")
    ENDIF()
  ENDIF()

  IF(NOT DEFINED _shared_input_file_list)
    SET(_shared_input_file_list "")
  ENDIF()

  IF(NOT DEFINED _input_file_list)
    IF(_current_dir_name MATCHES "^module_")
      SET(_input_file_list "")
    ELSE()
      SET(_input_file_list "input.prm")
    ENDIF()
  ENDIF()

  IF(NOT DEFINED _output_file)
    IF(_current_dir_name MATCHES "^module_")
      SET(_output_file output.out)
    ELSE()
      SET(_output_file iter_history.out)
    ENDIF()
  ENDIF()

  IF(NOT DEFINED _comparison_file)
    IF(_current_dir_name MATCHES "^module_")
      SET(_comparison_file output.out.reference)
    ELSE()
      SET(_comparison_file iter_history.out.reference)
    ENDIF()
  ENDIF()

  IF(NOT DEFINED _para_run)
    SET (_para_run ${_input_file_list} > screen.log 2>&1)
    LIST (APPEND _additional_clean_up_files "screen.log")
  ENDIF()
  UNSET(_current_dir_name)

  ENABLE_TESTING()
  PICKUP_ALL_SUBDIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
ENDMACRO()