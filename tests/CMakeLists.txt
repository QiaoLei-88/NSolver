
INCLUDE(${CMAKE_SOURCE_DIR}/utilities/cmake/setup_test_case.macro.cmake)

SET (_exec_copy ln )
SET (_para_copy -s )

SET (_exec_mpi mpirun)

# Find and set diff executable. Use numdiff if it exits, otherwise use standard diff
# instead. If non of them exits, abort building.
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
  IF( _exec_diff MATCHES "-NOTFOUND")
    MESSAGE(FATAL_ERROR "heheh"
    "Could not find diff or numdiff. One of those are required for running the testsuite.\n"
    "Please specify DIFF_DIR or NUMDIFF_DIR to a location containing the binaries."
    )  
  ENDIF()
  # numdiff not found but diff found
  SET (_para_diff "" )
ELSE()
  # numdiff found
  SET (_para_diff  -a 1e-8 -r 1e-8 -s  "' \\t\\n:<>=,\;'")
ENDIF()
# End find and set diff

SET (_exec_run ${CMAKE_BINARY_DIR}/bin/${PRJ_NAME})

SET (_time_out_time 18000)

PICKUP_ALL_SUBDIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
