add_executable(boost_asio boost_asio.cpp)
target_link_libraries(boost_asio PRIVATE Boost::asio)
target_link_libraries(boost_asio PRIVATE fmt::fmt-header-only)

add_executable(boost_threadpool boost_threadpool.cpp)
target_link_libraries(boost_threadpool PRIVATE Boost::system Boost::asio)

add_executable(boost_uuid boost_uuid.cpp)
target_link_libraries(boost_uuid PRIVATE Boost::system)
