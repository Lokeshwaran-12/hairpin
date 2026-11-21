"""
Utility Script: Cleans Python bytecode cache and transient runtime caches.
Usage: python scripts/clean_cache.py
"""
import os
import shutil
from pathlib import Path


def clean_project_caches():
    root_dir = Path(__file__).resolve().parent.parent
    print(f"Cleaning cache files in: {root_dir}")

    deleted_dirs = 0
    deleted_files = 0

    # 1. Remove __pycache__ directories
    for pycache in root_dir.rglob("__pycache__"):
        if "node_modules" not in str(pycache) and ".git" not in str(pycache):
            try:
                shutil.rmtree(pycache)
                deleted_dirs += 1
            except Exception as e:
                print(f"Warning: Could not remove {pycache}: {e}")

    # 2. Remove .pyc / .pyo files
    for pyc in root_dir.rglob("*.py[co]"):
        if "node_modules" not in str(pyc) and ".git" not in str(pyc):
            try:
                pyc.unlink()
                deleted_files += 1
            except Exception as e:
                print(f"Warning: Could not remove {pyc}: {e}")

    # 3. Clean empty transient upload directories if empty
    uploaded_dir = root_dir / "data" / "uploaded_videos"
    if uploaded_dir.exists():
        for f in uploaded_dir.glob("*"):
            if f.is_file() and f.name != ".gitkeep":
                try:
                    f.unlink()
                    deleted_files += 1
                except Exception:
                    pass

    print(f"Done: Removed {deleted_dirs} cache directories and {deleted_files} transient cache files.")


if __name__ == "__main__":
    clean_project_caches()
