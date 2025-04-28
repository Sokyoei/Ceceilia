set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")

macro(install_ceceilia_exe exe)
    install(TARGETS ${exe} RUNTIME DESTINATION bin)
endmacro(install_ceceilia_exe)

macro(install_ceceilia_lib lib)
    install(
        TARGETS ${lib}
        EXPORT ${PROJECT_NAME}Targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
    )
endmacro(install_ceceilia_lib)
