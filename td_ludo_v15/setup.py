"""Build script for td_ludo_v15_cpp — V15's fresh cell-based engine.

Mirrors td_ludo/setup.py exactly except:
  - extension name is `td_ludo_v15_cpp` (parallel module, no symbol collision)
  - sources are V15-only: src/bindings_v15.cpp, src/game_v15.cpp
  - no MCTS (stripped from V15 — pure policy net, no tree search)

Build: `pip install -e .` from this directory.
"""
from setuptools import setup, Extension, find_packages
import pybind11

include_dirs = [pybind11.get_include(), "src"]

ext_modules = [
    Extension(
        "td_ludo_v15_cpp",
        ["src/bindings_v15.cpp", "src/game_v15.cpp"],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=["-std=c++14", "-O3"],
    ),
]

setup(
    name="td_ludo_v15",
    version="0.0.1",
    author="AlphaLudo Team",
    description="V15 — Graph Transformer + per-cell triplet encoder + fresh cell-based engine",
    packages=find_packages(),
    ext_modules=ext_modules,
    python_requires=">=3.10",
)
