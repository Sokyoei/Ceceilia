########################################################################################################################
# post install
########################################################################################################################
if(TARGET ${PROJECT_NAME})
    install(DIRECTORY include/ DESTINATION include)

    install(
        EXPORT ${PROJECT_NAME}Targets
        FILE ${PROJECT_NAME}Targets.cmake
        # NAMESPACE ${PROJECT_NAME}::
        NAMESPACE Ahri::
        DESTINATION ${CMAKE_INSTALL_LIB_DIR}/cmake/${PROJECT_NAME}
    )
endif()

install(
    RUNTIME_DEPENDENCY_SET runtime_deps
    PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-"
    POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
    RUNTIME DESTINATION bin
)

########################################################################################################################
# cpack
########################################################################################################################
if(WIN32)
    set(CPACK_GENERATOR "ZIP")
else()
    set(CPACK_GENERATOR "TGZ")
endif()

include(CPack)
set(CPACK_PACKAGE_NAME ${CMAKE_PROJECT_NAME})
set(CPACK_PACKAGE_VERSION ${CMAKE_PROJECT_VERSION})
set(CPACK_PACKAGE_CONTACT "Sokyoei@Ahri.com")
set(CPACK_PACKAGE_VENDOR "Ahri&Sokyoei&Nono")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Sokyoei's C/C++/CUDA Project")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
