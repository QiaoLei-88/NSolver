
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR}/header)

SET (_source_file_list "")

LIST(APPEND _source_file_list NSolver.cpp)
LIST(APPEND _source_file_list NSEquation.cpp)
LIST(APPEND _source_file_list AllParameters.cpp)
LIST(APPEND _source_file_list main.cpp)
LIST(APPEND _source_file_list MMS.cpp)
LIST(APPEND _source_file_list FEParameters.cpp)
LIST(APPEND _source_file_list Postprocessor.cpp)
LIST(APPEND _source_file_list WallForce.cpp)
LIST(APPEND _source_file_list NSolver_integrate_force.cpp)
LIST(APPEND _source_file_list PhysicalParameters.cpp)
LIST(APPEND _source_file_list EquationComponents.cpp)
LIST(APPEND _source_file_list TimeSteppingParameters.cpp)

ADD_EXECUTABLE(${PRJ_NAME}
  ${_source_file_list})

SET_PROPERTY(TARGET ${PRJ_NAME}
  APPEND PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )

DEAL_II_SETUP_TARGET(${PRJ_NAME})
