from pathlib import Path
import platform

ROOT = str(Path.cwd())
if platform.system() == "Windows":
    ROOT = ROOT.replace("\\", "/")


vars: list = [
    f'#define ROOT "{ROOT}"\n',
    '#include "include/Sokyoei.h"\n',
]

print(vars)


def write_config_h():
    with open("config.h", "w") as f:
        f.writelines(vars)


def main():
    write_config_h()


if __name__ == "__main__":
    main()
