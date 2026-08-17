add_executable(boost_asio boost_asio.cpp)
target_link_libraries(boost_asio PRIVATE Boost::asio)
target_link_libraries(boost_asio PRIVATE fmt::fmt-header-only)
