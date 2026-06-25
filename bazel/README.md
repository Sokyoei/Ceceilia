# Bazel

## 安装

使用 [bazelisk](https://github.com/bazelbuild/bazelisk#installation) 来管理 Bazel 版本。

| 工具       | 作用                  |
| :--------- | :-------------------- |
| bazel      | 核心构建与测试系统    |
| bazelisk   | bazel 版本管理器      |
| buildifier | 构建文件格式化和 Lint |
| buildozer  | 构建文件批量编辑工具  |

## 常用命令

```shell
# bazelisk 下载 bazel
bazel shutdown
# bazel 清理缓存
bazel clean --expunge
# bazel 查看 JDK
bazel info java-home
# Visual Studio 需要安装英文语言包
bazel sync --configure
```

## FAQ

### 1. PKIX path building failed

=== "Windows"

    ```shell
    cd $JAVA_HOME/bin
    ./keytool -importcert -alias your_alias -keystore ../lib/security/cacerts -file your_certificate.cer
    ```

=== "Linux"

    ```shell
    # /etc/ssl/certs/java/cacerts 是 Debian/Ubuntu 系统安装的 openjdk 的默认 cacerts 文件路径
    sudo keytool -importcert -alias your_alias -keystore /etc/ssl/certs/java/cacerts -file your_certificate.cer
    ```
