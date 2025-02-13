add_executable(boost_uuid boost_uuid.cpp)
target_link_libraries(boost_uuid PRIVATE Boost::system)
