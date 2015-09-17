# Parameters of macro are not variables.
# But parameter expansion is not verbatim text replacement, but some thing
# like this: ${curdir} -> value passed to macro parameter curdir.
MACRO(LIST_ALL_SUBDIRECTORIES result curdir)

  FILE(GLOB __children RELATIVE ${curdir} ${curdir}/*)

  SET(__dir_list "")
  FOREACH(__child ${__children})
    IF(IS_DIRECTORY ${curdir}/${__child})
      LIST(APPEND __dir_list ${__child})
      LIST(SORT __dir_list)
    ENDIF()
  ENDFOREACH()
  UNSET(__child)

  SET(${result} ${__dir_list})
  UNSET(__dir_list)
  
ENDMACRO()


MACRO(PICKUP_ALL_SUBDIRECTORIES curdir)

  SET(__sub_dir_list "")
  LIST_ALL_SUBDIRECTORIES(__sub_dir_list ${curdir})

  FOREACH(__sub_dir ${__sub_dir_list})
    ADD_SUBDIRECTORY(${__sub_dir})
  ENDFOREACH()

   UNSET(__sub_dir)
   UNSET(__sub_dir_list)

  ENABLE_TESTING()

ENDMACRO()
