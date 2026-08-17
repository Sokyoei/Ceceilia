# Linux

[Ubuntu](Ubuntu.md)
[CentOS](CentOS.md)

## Linux 文件权限

drwxrwxrwx

|      |              |
| :--- | :----------- |
| d    | 文件夹       |
| -    | 普通文件     |
| l    | 链接文件     |
| b    | 块设备文件   |
| p    | 管道文件     |
| c    | 字符设备文件 |
| s    | 套接字文件   |

## Linux 标准输入、标准输出和标准错误输出

|      |                      |
| :--- | :------------------- |
| 0    | 标准输入（键盘）     |
| 1    | 标准输出（屏幕）     |
| 2    | 标准错误输出（屏幕） |

## Linux '>', '<', '&', '|'

|      |                                                            |
| :--- | :--------------------------------------------------------- |
| >    | 输出重定向（覆盖）                                         |
| >>   | 输出重定向（追加）                                         |
| &    | 任务在后台执行                                             |
| &&   | 表示前一条命令执行成功，才执行下一条命令                   |
| \|   | 管道运算符，上一条命令的输出，作为下一条命令的输入（参数） |
| \|\| | 表示前一条命令执行失败，才执行下一条命令                   |
| &>   | 将错误信息和普通信息都重定向输出                           |

将 shell 里的文本写入文件

```shell
cat > helloworld.txt << EOF
"hello world"
other strings...
EOF
```

## Linux 命令

1. 系统信息

    - [uname](#uname)
    - hostname
    - uptime
    - lscpu

2. 文件目录

    - [ls](#ls) 列出文件
    - [mv](#mv) 移动/重命名文件
    - [rm](#rm) 删除文件/文件夹
    - [cp](#cp) 复制文件/文件夹
    - [ln](#ln) 创建链接
    - pwd 打印当前目录
    - [mkdir](#mkdir) 创建文件夹
    - touch 创建文件
    - cat/more/less 查看文件
    - head/tail 查看文件开头/结尾

3. 文件搜索
    - find
    - [which](#which)
    - [whereis](#whereis)

4. 文本处理
    - sed
    - awk
    - grep
    - sort
    - diff
    - wc
5. 文件压缩
    - tar 将一系列文件或文件夹打包归档
    - zip/unzip 压缩或解压文件 zip
    - gzip/gunzip 压缩或解压文件 gz
6. 权限管理
    - chmod 修改文件或文件夹的权限
    - chown
    - umask
7. 进程管理
    - ps
    - top 显示 linux 进程
    - htop
    - kill 杀死进程
    - killall
    - nohup 将命令在后台运行
8. 磁盘
    - du
    - df
    - mount/unmount 挂载/卸载磁盘
9. 安全
    - scp

10. 下载文件
    - wget
    - curl
11. 网络配置
    - ifconfig
    - ip
    - netstat
    - ping

12. ？？
    - ldd 打印动态库依赖

### ls

ls(list directory contents) 列出文件信息

```shell
# 查看包含隐藏文件
ls -a
# 查看文件详细信息
ls -l
# 查看文件详细信息（文件大小美化显示）
ls -lh
```

### mv

mv(move) 移动或重命名文件或文件夹

```shell
# 移动（不显示确认与否）
mv -f src dst
```

### pwd

pwd(print the name of the current working directory) 打印当前工作路径

```shell
pwd
```

### cp

cp(copy)

```shell
cp src dst
```

### rm

rm(remove)

```shell
# 删除文件
rm file
# 删除文件夹
rm -rf dir
```

### ln

ln(link)

### mkdir

mkdir(make directory)

### chmod

chmod(change mode)

```shell
chmod 777 file
```

### touch

### vi

### vim

### less

### tail

```shell
# 一直滚动输出
tail -f txt_file
```

### cat

显示文本文件的内容

```shell
cat txt_file
```

### awk

处理文本的应用程序

### sed

对文本进行过滤和变形处理

### file

查看文件类型

### gzip

### nohup

### kill

终止进程

### killall

终止一系列进程

### uname

显示内核信息

### free

查看内存使用情况

### reboot

重启

### shutdown

关机

### find

查找文件的位置

### grep

对文件内容搜索，返回所有匹配的行

### which

输出命令的全路径

```shell
which exe_file
```

### whereis

显示某个命令的位置

### dd

复制磁盘或文件系统

### clear

清空屏幕

### du

du(disk usage) 查看磁盘使用率

```shell
du -sh
```

### df

df(disk free)

```shell
df -h
```

### systemctl

```shell
# 查看开机启动项
systemctl list-unit-files --type=service
```

### ldd

```shell
# 查看可执行文件的依赖库
ldd exe_file
```

### top

top 性能分析工具（类似 Windows 任务管理器）

```shell
top
# 使用后进入交互模式
```

|      |          |
| :--- | :------- |
| q    | 退出     |
| h    | 帮助界面 |

### ps

ps 查看进程、应用运行状态

```shell
# 查看应用的进程
ps -ef | grep exe
```

### mount

mount

```shell
```

### ulimit

```shell
ulimit -s unlimited
```

## 关键文件

1. /etc/rc.d/rc.local 开机自启动一些指令
2. ~/.bash_profile 设置个人用户的环境变量、个人设置等
3. /etc/profile 设置系统级的环境变量、启动程序等，对所有用户生效
4. ~/.bashrc 用来配置 shell
