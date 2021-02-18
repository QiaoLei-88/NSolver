
INCLUDE(${CMAKE_SOURCE_DIR}/utilities/cmake/setup_test_case.macro.cmake)
INCLUDE(${CMAKE_SOURCE_DIR}/utilities/cmake/setup_module_test_case.macro.cmake)
INCLUDE(${CMAKE_SOURCE_DIR}/utilities/cmake/setup_test_category.macro.cmake)

SET (_exec_copy ln )
SET (_para_copy -s )

SET (_exec_mpi mpirun)

# Find and set diff executable. Use numdiff if it exits, otherwise use standard
# diff instead. If non of them exits, abort building.
# Note: FIND_PROGRAM will nor run when '_exec_diff' is defined.
FIND_PROGRAM(_exec_diff
  NAMES numdiff
  HINTS ${NUMDIFF_DIR}
  PATH_SUFFIXES bin
  )
IF( _exec_diff MATCHES "-NOTFOUND")
  FIND_PROGRAM(_exec_diff
    NAMES diff
    HINTS ${DIFF_DIR}
    PATH_SUFFIXES bin
    )
ENDIF()

IF( _exec_diff MATCHES "-NOTFOUND")
  MESSAGE(FATAL_ERROR "FATAL ERROR:\n"
  "Could not find diff or numdiff. One of those are required for running the testsuite.\n"
  "Please specify DIFF_DIR or NUMDIFF_DIR to the path of comparison tools."
  )
ENDIF()

IF( _exec_diff MATCHES "numdiff")
  MESSAGE(STATUS "Finding comparison tool: numdiff found at ${_exec_diff}")
  SET (_para_diff " ")
ELSEIF( _exec_diff MATCHES "diff")
  MESSAGE(STATUS "Finding comparison tool: diff found at ${_exec_diff}")
  SET (_para_diff "")
ELSE()
  MESSAGE(FATAL_ERROR "FATAL ERROR:\n"
  "Unknown value for comparison tool.\n"
  )
ENDIF()
# End find and set diff

SET (_exec_run ${CMAKE_BINARY_DIR}/bin/${PRJ_NAME})

SET (_time_out_time 18000)

# Set up static library list for module tests needs to build individual executables
LIST_ALL_SUBDIRECTORIES (_static_lib_list ${CMAKE_SOURCE_DIR}/source)

SET (_static_lib_for_modele_test "")
FOREACH(__sub_dir ${_static_lib_list})
  add_library(${__sub_dir}_for_modele_test STATIC IMPORTED)
  set_property(TARGET ${__sub_dir}_for_modele_test PROPERTY IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/source/${__sub_dir}/lib${__sub_dir}.a)
  LIST(APPEND _static_lib_for_modele_test ${__sub_dir}_for_modele_test)
ENDFOREACH()
UNSET(__sub_dir)
SET(_addtional_solver_outputs *.vtu *.pvtu *_runtime.error run.success time_history.out paper_data.txt boundaryData.slot-*.raw)

# Set up clean target
LIST_ALL_SUBDIRECTORIES (__sub_dir_list ${CMAKE_CURRENT_SOURCE_DIR})
FOREACH(__sub_dir ${__sub_dir_list})
  ADD_CUSTOM_TARGET(clean_${__sub_dir})
  ADD_DEPENDENCIES(clean_all_tests clean_${__sub_dir})
ENDFOREACH()
UNSET(__sub_dir)
UNSET(__sub_dir_list)

PICKUP_ALL_SUBDIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
