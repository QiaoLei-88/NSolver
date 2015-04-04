
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
IF(NOT DEAL_II_WITH_TRILINOS)
  MESSAGE(FATAL_ERROR "\n"
    "Error! The deal.II library found at ${DEAL_II_PATH} was not configured with"
    "    DEAL_II_WITH_TRILINOS = ON"
    "One or all of these are OFF in your installation but are required for this tutorial step."
    )
ENDIF()

#
# Let's move
#
INCLUDE(${CMAKE_SOURCE_DIR}/utilities/cmake/list_all_subdirectories.macro.cmake)
SET(PRJ_NAME "NSolver")


DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${PRJ_NAME})

ENABLE_TESTING()

ADD_SUBDIRECTORY(source)
ADD_SUBDIRECTORY(tests)
