# RIME

## 安装 Flatpak RIME 中文输入法

### 1. 删除 apt 安装的 Fcitx5

```shell
sudo apt remove fcitx5 -y
sudo apt autoremove

# 清理残留配置
rm -rf ~/.local/share/fcitx5
rm -rf ~/.config/fcitx5
```

### 2. 安装 Flatpak Fcitx5

```shell
sudo apt update
sudo apt install flatpak gnome-software-plugin-flatpak -y

# 添加 Flatpak 源
flatpak remote-add --if-not-exists flathub https://dl.flathub.org/repo/flathub.flatpakrepo

flatpak install flathub org.fcitx.Fcitx5 -y
flatpak install flathub org.fcitx.Fcitx5.Addon.Rime -y
flatpak install flathub org.fcitx.Fcitx5.Addon.ChineseAddons -y
```

### 3. 安装`雾凇拼音`

```shell
mkdir -p ~/.var/app/org.fcitx.Fcitx5/data/fcitx5
cd ~/.var/app/org.fcitx.Fcitx5/data/fcitx5
git clone https://github.com/iDvel/rime-ice.git rime
```

### 4. 环境变量配置

```shell
sudo tee -a /etc/environment <<'EOF'
GTK_IM_MODULE=fcitx
QT_IM_MODULE=fcitx
XMODIFIERS=@im=fcitx
SDL_IM_MODULE=fcitx
EOF
```

查看是否写入成功，重启后环境变量才能配置成功。

```shell
cat /etc/environment
```

### 5. 主题配置

```shell
mkdir -p ~/.var/app/org.fcitx.Fcitx5/data/fcitx5/themes/
cd ~/.var/app/org.fcitx.Fcitx5/data/fcitx5/themes/

# 我使用 catppuccin 主题
git clone https://github.com/catppuccin/fcitx5 catppuccin
cp -r catppuccin/src/* ./
```

### 6. 实用工具和更新

打开配置面板（右上角点击输入法设置也行）

```shell
flatpak run --command=fcitx5-configtool org.fcitx.Fcitx5
```

更新 rime

```shell
cd ~/.var/app/org.fcitx.Fcitx5/data/fcitx5/rime/
git pull
# 重载 rime 部署
flatpak run --command=fcitx5-remote org.fcitx.Fcitx5 -r
```

###### 参考

[【输入法】Ubuntu 22.04 终极输入法方案：Fcitx5 + 雾凇拼音 (Flatpak版)](https://blog.csdn.net/xuzhengzhe/article/details/156383333)
