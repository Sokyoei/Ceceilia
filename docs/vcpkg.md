# vcpkg

## 集成到 Visual Studio

```shell
vcpkg integrate intall
# 卸载
vcpkg integrate remove
```

## 清单模式

增加当前基线到 vcpkg.json

```shell
vcpkg x-update-baseline --add-initial-baseline
```

查看库的基线

```shell
git log "--format=%H %cd %s" --date=short --left-only -- versions/b-/boost.json
```
