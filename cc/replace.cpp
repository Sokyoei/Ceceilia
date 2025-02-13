/**
 * @file replace.cpp
 * @date 2025/02/07
 * @author Sokyoei
 * @details
 * C++ keyword replace C++ 关键字替换
 * +------+----------+
 * |  &&  |  and     |
 * |  &=  |  and_eq  |
 * |  ||  |  or      |
 * |  |=  |  or_eq   |
 * |  !   |  not     |
 * |  !=  |  not_eq  |
 * |  ^   |  xor     |
 * |  ^=  |  xor_eq  |
 * |  &   |  bitand  |
 * |  |   |  bitor   |
 * |  ~   |  compl   |
 * |  {   |  <%      |
 * |  }   |  %>      |
 * |  [   |  <:      |
 * |  ]   |  :>      |
 * |  #   |  %:      |
 * |  ##  |  %:%:    |
 * +------+----------+
 */

// clang-format off
%:include <iostream>
%:include <bitset>

int main(int argc, char* argv<::>) <%
    std::bitset<4> bit(1100);
    bit = compl bit;
    std::cout << bit << std::endl;
    return 0;
%>
// clang-format on
