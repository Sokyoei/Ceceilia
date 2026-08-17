import platform
from pathlib import Path

CECEILIA_ROOT = str(Path.cwd())
if platform.system() == "Windows":
    CECEILIA_ROOT = CECEILIA_ROOT.replace("\\", "/")


vars: list = [
    f'#define CECEILIA_ROOT "{CECEILIA_ROOT}"\n',
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
