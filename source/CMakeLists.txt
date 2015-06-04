
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR}/header)

SET (_source_file_list "")
SET (_static_lib_list "")

LIST(APPEND _source_file_list NSEquation.cpp)
LIST(APPEND _source_file_list AllParameters.cpp)
LIST(APPEND _source_file_list main.cpp)
LIST(APPEND _source_file_list MMS.cpp)
LIST(APPEND _source_file_list FEParameters.cpp)
LIST(APPEND _source_file_list Postprocessor.cpp)
LIST(APPEND _source_file_list WallForce.cpp)
LIST(APPEND _source_file_list PhysicalParameters.cpp)
LIST(APPEND _source_file_list EquationComponents.cpp)
LIST(APPEND _source_file_list TimeSteppingParameters.cpp)

# Pick up all subdirectories for static libraries
LIST_ALL_SUBDIRECTORIES (_static_lib_list ${CMAKE_CURRENT_SOURCE_DIR})
FOREACH(__sub_dir ${_static_lib_list})
  ADD_SUBDIRECTORY(${__sub_dir})
ENDFOREACH()
UNSET(__sub_dir)

# Set up target for executable
ADD_EXECUTABLE(${PRJ_NAME}
  ${_source_file_list})

TARGET_LINK_LIBRARIES(${PRJ_NAME} 
  ${_static_lib_list})

SET_PROPERTY(TARGET ${PRJ_NAME}
  APPEND PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )

DEAL_II_SETUP_TARGET(${PRJ_NAME})
