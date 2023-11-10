#include <filesystem>
#include <iostream>

#include <tinyxml2.h>

#include "config.h"

namespace Ahri {
/**
 * @brief 递归遍历 XML 的元素
 * @param element XML 根节点
 */
void print_xml(tinyxml2::XMLElement* root) {
    for (auto current_element = root->FirstChildElement(); current_element;
         current_element = current_element->NextSiblingElement()) {
        auto temp_element = current_element;
        std::cout << temp_element->Name() << std::endl;
        if (auto en_us = current_element->Attribute("en-us")) {
            std::cout << en_us << std::endl;
        }
        if (auto zh_cn = current_element->Attribute("zh-cn")) {
            std::cout << zh_cn << std::endl;
        }
        if (auto zh_tw = current_element->Attribute("zh-tw")) {
            std::cout << zh_tw << std::endl;
        }
        if (auto ko_kr = current_element->Attribute("ko-kr")) {
            std::cout << ko_kr << std::endl;
        }
        if (auto ja_jp = current_element->Attribute("ja-jp")) {
            std::cout << ja_jp << std::endl;
        }
        if (!temp_element->NoChildren()) {
            print_xml(temp_element);
        }
    }
}
}  // namespace Ahri

int main(int argc, char* argv[]) {
#ifdef _WIN32
    system("chcp 65001");  // Windows 终端修改代码页以显示 UTF8 字符
#endif
    auto xml_file = std::filesystem::path(ROOT) / "data/Ahri/Ahri.xml";
    tinyxml2::XMLDocument xml;
    auto error = xml.LoadFile(xml_file.string().c_str());
    if (error == tinyxml2::XML_SUCCESS) {
        auto root = xml.RootElement();
        Ahri::print_xml(root);
    }
    return 0;
}
