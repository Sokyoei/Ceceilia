aux_source_directory(controller CONTROLLER_SRCS)
aux_source_directory(models MODELS_SRCS)
aux_source_directory(services SERVICES_SRCS)

add_executable(drogon_main
    main.cpp
    ${CONTROLLER_SRCS}
    ${MODELS_SRCS}
    ${SERVICES_SRCS}
)
target_link_libraries(drogon_main PRIVATE Drogon::Drogon)
