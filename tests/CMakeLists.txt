
INCLUDE(${CMAKE_SOURCE_DIR}/utilities/cmake/setup_test_case.macro.cmake)

SET (_exec_copy ln )
SET (_para_copy -s )

SET (_exec_diff diff )
SET (_para_diff "" )

SET (_exec_run ${CMAKE_BINARY_DIR}/bin/${PRJ_NAME})

PICKUP_ALL_SUBDIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
