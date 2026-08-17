# CentOS

## 安装 Chrome

```shell
wget https://dl.google.com/linux/direct/google-chrome-stable_current_x86_64.rpm
yum install -y google-chrome-stable_current_x86_64.rpm
```

将图标拖到桌面上右击 Properties, 在 Command 命令行增加 -no-sandbox

```shell
/usr/bin/google-chrome-stable %U --no-sandbox
```
