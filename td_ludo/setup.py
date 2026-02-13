from setuptools import setup, Extension
import pybind11
import os

# Available at setup time due to pyproject.toml or manual install
include_dirs = [pybind11.get_include(), "src"]

ext_modules = [
    Extension(
        "td_ludo_cpp",
        ["src/bindings.cpp", "src/game.cpp", "src/mcts.cpp"],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=["-std=c++14", "-O3"],
    ),
]

setup(
    name="td_ludo_cpp",
    version="0.0.1",
    author="AlphaLudo Team",
    description="High-performance Ludo engine in C++ (Isolated for TD Learning)",
    ext_modules=ext_modules,
)
