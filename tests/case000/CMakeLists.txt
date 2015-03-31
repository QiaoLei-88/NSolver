

ADD_CUSTOM_TARGET(run_test_case_000
	COMMAND ln -s ${CMAKE_CURRENT_SOURCE_DIR}/*.* .
	COMMAND ${CMAKE_BINARY_DIR}/bin/${PRJ_NAME} input.prm.2412 2>&1 > screen.log
	)

ENABLE_TESTING()
ADD_TEST(NAME check_test_case_000
	COMMAND diff screen.log screen.log.reference
	)
