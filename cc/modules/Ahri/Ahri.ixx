module;

export module Ahri;

// 导入 Ahri 的 KDA 和 Popstar 模块分区
import :KDA;
import :Popstar;

// 导入导出子模块
export import Ahri.Nono;
export import Ahri.Sokyoei;

export namespace Ahri {
int func(int x) {
    return add(x, 1) + sub<int>(x, 1);
}
}  // namespace Ahri
