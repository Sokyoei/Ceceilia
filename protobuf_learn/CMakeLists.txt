set(
    PROTO_FILES
    "protobuf_v2_learn.proto"
    "protobuf_v3_learn.proto"
)

foreach(PROTO_FILE ${PROTO_FILES})
    get_filename_component(PROTO_FILE_NAME ${PROTO_FILE} NAME)
    get_filename_component(PROTO_FILE_DIR ${PROTO_FILE} DIRECTORY)
    execute_process(
        COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
        --cpp_out=${CMAKE_CURRENT_SOURCE_DIR}/${PROTO_FILE_DIR}
        --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/${PROTO_FILE_DIR}
        ${PROTO_FILE_NAME}
    )
endforeach(PROTO_FILE ${PROTO_FILES})

add_executable(protobuf_learn protobuf_learn.cpp)
target_link_libraries(protobuf_learn PRIVATE protobuf::libprotoc protobuf::libprotobuf protobuf::libprotobuf-lite)
