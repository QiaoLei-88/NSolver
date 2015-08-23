
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.1 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

#
# Are all dependencies fullfilled?
#
SET (_depend_features TRILINOS P4EST MPI) 
FOREACH (_feature ${_depend_features})
IF(NOT DEAL_II_WITH_${_feature})
  MESSAGE(FATAL_ERROR "\n"
    "Error! The deal.II library found at ${DEAL_II_PATH} was not configured with\n"
    "    DEAL_II_WITH_${_feature}=ON\n"
    "This feature is set to OFF in your installation but are required for this program step.\n"
    )
ENDIF()
ENDFOREACH(_feature)
UNSET(_feature)
UNSET(_depend_features)

#
# Never build in source directory
#
IF("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  MESSAGE(FATAL_ERROR "\n"
    "  Do not build inside source directory!\n"
    "  Set up a dedicated build directory and run\n"
    "      ${CMAKE_COMMAND} ${CMAKE_SOURCE_DIR}\n"
    "  from that directory.\n\n"
    )
ENDIF()

#
# Let's move
#
INCLUDE(${CMAKE_SOURCE_DIR}/utilities/cmake/list_all_subdirectories.macro.cmake)
SET(PRJ_NAME "NSolver")

INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR}/header)
DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${PRJ_NAME})


ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
  )

# Define a distclean target to remove every generated file:
ADD_CUSTOM_TARGET(distclean
  COMMAND ${CMAKE_COMMAND} -E remove_directory CMakeFiles
  COMMAND ${CMAKE_COMMAND} -E remove_directory Testing
  COMMAND ${CMAKE_COMMAND} -E remove_directory bin
  COMMAND ${CMAKE_COMMAND} -E remove_directory source
  COMMAND ${CMAKE_COMMAND} -E remove_directory tests
  COMMAND ${CMAKE_COMMAND} -E remove CMakeCache.txt cmake_install.cmake Makefile CTestTestfile.cmake
  COMMENT "distclean invoked"
  )

# Define a reset target to regenerate the cmake files:
# This target just do distclean plus ${CMAKE_COMMAND} ${CMAKE_SOURCE_DIR}, but we 
# cannot make it depending on target distclean because after distclean there will 
# be no rule to make target reset.
ADD_CUSTOM_TARGET(reset
  COMMAND ${CMAKE_COMMAND} -E remove_directory CMakeFiles
  COMMAND ${CMAKE_COMMAND} -E remove_directory Testing
  COMMAND ${CMAKE_COMMAND} -E remove_directory bin
  COMMAND ${CMAKE_COMMAND} -E remove_directory source
  COMMAND ${CMAKE_COMMAND} -E remove_directory tests
  COMMAND ${CMAKE_COMMAND} -E remove CMakeCache.txt cmake_install.cmake Makefile CTestTestfile.cmake
  COMMAND ${CMAKE_COMMAND} ${CMAKE_SOURCE_DIR}
  COMMENT "reset invoked"
  )


ADD_CUSTOM_TARGET(info
  COMMAND ${CMAKE_COMMAND}
    -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -P ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/print_usage.cmake
  )

ENABLE_TESTING()

ADD_SUBDIRECTORY(source)
ADD_SUBDIRECTORY(tests)

# Print help message once:
IF(CMAKE_GENERATOR MATCHES "Ninja")
  SET(_make_command "$ ninja")
ELSE()
  SET(_make_command "$  make")
ENDIF()

IF(NOT USAGE_PRINTED)
  CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/utilities/cmake/print_usage.cmake.in  ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/print_usage.cmake ESCAPE_QUOTES @ONLY)
  INCLUDE(${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/print_usage.cmake)
  SET(USAGE_PRINTED TRUE CACHE INTERNAL "")
ELSE()
  MESSAGE(STATUS "\n\nRun  ${_make_command} info  to print a detailed help message.\n\n")
ENDIF()
