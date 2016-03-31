MACRO(SETUP_MODULE_TEST_CASE)

# Set a target for the test driver executable, the follow the working flow of
# SETUP_TEST_CASE()

GET_FILENAME_COMPONENT(_current_dir_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
SET (_test_name ${_test_name_prefix}_${_current_dir_name})
UNSET(_current_dir_name)

SET (__test_driver  ${_test_name}_exe)
IF (_exec_run MATCHES "${_exec_mpi}")
    SET (_exec_run ${_exec_mpi})
    IF(NOT DEFINED _n_slots)
      SET(_n_slots 2)
    ENDIF()
    SET (_para_run -n ${_n_slots} ${CMAKE_CURRENT_BINARY_DIR}/${__test_driver} ${_input_file_list} > screen.log 2>&1)
    LIST (APPEND _additional_clean_up_files "screen.log")
ELSE()
  SET (_exec_run ${__test_driver})
ENDIF()
set (_extra_depends_for_output ${__test_driver})
add_executable(${__test_driver}
               EXCLUDE_FROM_ALL
               ${_src_file_list})

TARGET_LINK_LIBRARIES(${__test_driver}
  ${_static_lib_for_modele_test}
  ${_static_lib_for_modele_test})

DEAL_II_SETUP_TARGET(${__test_driver})
SETUP_TEST_CASE()

ENDMACRO()
