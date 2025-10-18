#!/usr/bin/env python3
"""
download_data_zip.py
从指定 Google Drive 文件夹下载名为 data.zip 的文件到当前目录。
依赖: pip install gdown
"""

import os
import sys
import shutil
from pathlib import Path

def main():
    FOLDER_URL = "https://drive.google.com/drive/folders/1eewycbcAJ95atmF_V3bNchPIFDSw_TQC"
    TARGET_NAME = "data.zip"  # 你要的文件名
    TMP_DIR = Path("./_gdrive_tmp")

    try:
        import gdown
    except ImportError:
        print("未找到 gdown，请先运行: pip install gdown", file=sys.stderr)
        sys.exit(1)

    # 1) 下载整个文件夹到临时目录（gdown 会按子目录结构保存）
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Info] 同步文件夹到: {TMP_DIR.resolve()}")
    gdown.download_folder(
        url=FOLDER_URL,
        output=str(TMP_DIR),
        quiet=False,
        use_cookies=False
    )

    # 2) 在临时目录里查找名为 data.zip 的文件（大小写不敏感）
    target_lower = TARGET_NAME.lower()
    candidates = [p for p in TMP_DIR.rglob("*") if p.is_file() and p.name.lower() == target_lower]

    if not candidates:
        print(f"[Error] 未在文件夹中找到 {TARGET_NAME}。请确认文件名是否正确。", file=sys.stderr)
        sys.exit(2)

    # 如果同名文件有多个，默认取第一个（也可改成让用户选择）
    src = candidates[0]
    dst = Path("./") / TARGET_NAME

    # 3) 复制到当前目录（存在则覆盖）
    shutil.copy2(src, dst)
    print(f"[OK] 已保存到: {dst.resolve()}")

    # 4)（可选）清理临时目录
    # shutil.rmtree(TMP_DIR)

if __name__ == "__main__":
    main()
