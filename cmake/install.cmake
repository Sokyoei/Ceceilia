function(install_ceceilia)

endfunction(install_ceceilia)

install(
    EXPORT ${PROJECT_NAME}-config
    NAMESPACE Ahri::
    DESTINATION ${CMAKE_INSTALL_LIB_DIR}/cmake/${PROJECT_NAME}
)

install(
    DIRECTORY
    DESTINATION
)
