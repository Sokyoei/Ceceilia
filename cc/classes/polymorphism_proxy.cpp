#include <sstream>

#include <fmt/core.h>
#include <proxy/proxy.h>

namespace Ahri {
PRO_DEF_MEM_DISPATCH(MemDraw, draw);
PRO_DEF_MEM_DISPATCH(MemArea, area);

struct Drawable
    : pro::facade_builder ::add_convention<MemDraw, std::string()>::add_convention<MemArea, double() noexcept>::
          support_copy<pro::constraint_level::nontrivial>::build {};

class Rectangle {
public:
    Rectangle(double width, double height) : _width(width), _height(height) {}
    Rectangle(const Rectangle&) = default;

    std::string draw() const { return std::string(fmt::format("Rectangle: width = {}, height = {}", _width, _height)); }
    double area() const noexcept { return _width * _height; }

private:
    double _width;
    double _height;
};

void print_drawable(pro::proxy<Drawable> p) {
    fmt::println("entity is {}, area= {}.", p->draw(), p->area());
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    pro::proxy<Ahri::Drawable> p = pro::make_proxy<Ahri::Drawable, Ahri::Rectangle>(3, 5);
    Ahri::print_drawable(p);

    return 0;
}
