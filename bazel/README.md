# Bazel

## 安装

使用 [Bazelisk](https://github.com/bazelbuild/bazelisk) 来管理 Bazel 版本。

## Bazel 清理缓存

```shell
bazel clean --expunge
```

## Visual Studio 需要安装英文语言包

```shell
bazel sync --configure
```
