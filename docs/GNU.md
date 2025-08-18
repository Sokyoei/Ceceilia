# GNU

[docs](https://gcc.gnu.org/onlinedocs/)

|         |                                    |
| :------ | :--------------------------------- |
| gcc     | C 编译器                           |
| g++     | C++ 编译器                         |
| cpp     | C 预处理器(C PreProcessor)         |
| as      | asm 汇编器                         |
| ld      | 链接器                             |
| ar      | 创建、修改和提取档案文件(Archives) |
| nm      | 列出二进制文件的符号               |
| objdump | 列出二进制文件的各种信息、反汇编   |

## gcc/g++

|       |      |
| :---- | :--- |
| -Wall |      |
| -fPIC |      |

### -fPIC

PIC(Position-Independent Code), 位置无关代码

## gdb

|           |      |                          |
| :-------- | :--- | :----------------------- |
| break     | b    | 设置断点                 |
| run       |      | 运行程序到断点（如果有） |
| next      | n    | 单步执行                 |
| step      | s    | 单步步入                 |
| continue  | c    | 继续执行                 |
| finish    |      | 单步跳出                 |
| print     | p    |                          |
| info      |      |                          |
| backtrace | bt   | 打印调用堆栈             |
| attach    |      |                          |
| list      |      |                          |
|           |      |                          |

## FAQ

### 高版本 GCC 安装在底版本 linux 上时，编译连接报错 `GLIBCXX_X.Y.ZZ not found`

CMake 项目可以在 CMakePresets.json 文件中设置环境变量 LD_LIBRARY_PATH 来解决这个问题。

```json
        {
            "name": "GCC_15.1.0_x86_64-pc-linux-gnu",
            "displayName": "GCC 15.1.0 x86_64-pc-linux-gnu",
            "description": "正在使用编译器: C = /usr/local/bin/gcc-15.1, CXX = /usr/local/bin/g++-15.1",
            "binaryDir": "${sourceDir}/out/build/${presetName}",
            "cacheVariables": {
                "CMAKE_INSTALL_PREFIX": "${sourceDir}/out/install/${presetName}",
                "CMAKE_C_COMPILER": "/usr/local/bin/gcc-15.1",
                "CMAKE_CXX_COMPILER": "/usr/local/bin/g++-15.1",
                "CMAKE_BUILD_TYPE": "Debug"
            },
            "environment": {
                "LD_LIBRARY_PATH": "/usr/local/lib64:{env:LD_LIBRARY_PATH}"
            }
        }
```

也可以设置链接 Flags

```cmake
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath=/usr/local/lib64")
```
