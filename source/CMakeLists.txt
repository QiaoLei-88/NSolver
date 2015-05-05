
ADD_EXECUTABLE(${PRJ_NAME}
  NSolver.cpp
  NSEquation.cpp
  AllParameters.cpp
  main.cpp
  MMS.cpp
  FEParameters.cpp
  Postprocessor.cpp
  )

SET_PROPERTY(TARGET ${PRJ_NAME}
  APPEND PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )

DEAL_II_SETUP_TARGET(${PRJ_NAME})
