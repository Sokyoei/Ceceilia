add_executable(boost_threadpool boost_threadpool.cpp)
target_link_libraries(boost_threadpool PRIVATE Boost::system Boost::asio)

add_executable(boost_uuid boost_uuid.cpp)
target_link_libraries(boost_uuid PRIVATE Boost::system)
