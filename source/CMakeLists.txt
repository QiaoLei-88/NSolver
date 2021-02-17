
SET (_source_file_list "")
SET (_static_lib_list "")
FILE(GLOB _source_file_list *.cpp)


# Pick up all subdirectories for static libraries
LIST_ALL_SUBDIRECTORIES (_static_lib_list ${CMAKE_CURRENT_SOURCE_DIR})
# Instead add all subdirectories automatically via the above command,
# we have to do this in proper order manually. Because solver is depending on
# linearVelocityPotential and gcc cares the order.
#LIST(APPEND _static_lib_list solver)
#LIST(APPEND _static_lib_list Parameters)
#LIST(APPEND _static_lib_list linearVelocityPotential)

FOREACH(__sub_dir ${_static_lib_list})
  ADD_SUBDIRECTORY(${__sub_dir})
ENDFOREACH()
UNSET(__sub_dir)

# Set up target for executable
ADD_EXECUTABLE(${PRJ_NAME}
  ${_source_file_list})

# gcc only search for dependency in one direction during linking,
# so we add the library list twice to make sure gcc can always
# find what it want.
TARGET_LINK_LIBRARIES(${PRJ_NAME}
  ${_static_lib_list}
  ${_static_lib_list})

SET_PROPERTY(TARGET ${PRJ_NAME}
  APPEND PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )

ADD_CUSTOM_TARGET(
    VERSION_STAMP
    ${CMAKE_COMMAND} -D SRC=${CMAKE_SOURCE_DIR}/header/NSolver/version.info.in
                     -D DST=${CMAKE_BINARY_DIR}/version.info
                     -D GIT_EXEC=git
                     -D GIT_DIR=${CMAKE_SOURCE_DIR}
                     -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                     -P ${CMAKE_SOURCE_DIR}/utilities/cmake/version_stamp.cmake
)
INCLUDE_DIRECTORIES(AFTER ${CMAKE_BINARY_DIR})
ADD_DEPENDENCIES(${PRJ_NAME} VERSION_STAMP)

DEAL_II_SETUP_TARGET(${PRJ_NAME})
