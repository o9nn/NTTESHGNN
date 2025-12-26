#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ntteshgnn::ntteshgnn" for configuration ""
set_property(TARGET ntteshgnn::ntteshgnn APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(ntteshgnn::ntteshgnn PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libntteshgnn.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS ntteshgnn::ntteshgnn )
list(APPEND _IMPORT_CHECK_FILES_FOR_ntteshgnn::ntteshgnn "${_IMPORT_PREFIX}/lib/libntteshgnn.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
