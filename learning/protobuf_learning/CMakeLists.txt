set(
    PROTO_FILES
    "protobuf_v2.proto"
    "protobuf_v3.proto"
)

# build .proto file
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

add_executable(protobuf_main protobuf_main.cpp protobuf_v2.pb.cc protobuf_v2.pb.h)
target_link_libraries(protobuf_main PRIVATE protobuf::libprotoc protobuf::libprotobuf protobuf::libprotobuf-lite)
target_link_libraries(protobuf_main PRIVATE fmt::fmt-header-only)
