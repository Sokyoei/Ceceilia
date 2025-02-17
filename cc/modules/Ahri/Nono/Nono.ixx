module;

export module Ahri.Nono;

// export import "Animal.hpp";  // 全局导入并导出 Animal.hpp 中的内容
import "Animal.hpp";

export namespace Ahri::Nono {
using ::Animal;  // 将 Animal 引入到 Nono 命名空间中
}  // namespace Ahri::Nono
