"""
MLE Runtime - Python Client Library
Setup script
"""

from setuptools import setup, Extension, find_packages
import sys
import os

# Read version
version = "1.0.1"

# Read README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# C++ extension module
ext_modules = []
if not os.environ.get("MLE_SKIP_BUILD"):
    import platform
    
    # Platform-specific compiler flags
    if platform.system() == "Windows":
        extra_compile_args = ["/std:c++17", "/O2"]
    else:
        extra_compile_args = ["-std=c++17", "-O3"]
    
    ext_modules = [
        Extension(
            "mle_runtime._mle_core",
            sources=[
                "src/python_bindings.cpp",
                "../../cpp_core/src/engine.cpp",
                "../../cpp_core/src/loader.cpp",
                "../../cpp_core/src/executor.cpp",
                "../../cpp_core/src/ops_cpu.cpp",
            ],
            include_dirs=[
                "../../cpp_core/include",
            ],
            extra_compile_args=extra_compile_args,
            language="c++",
        )
    ]

setup(
    name="mle-runtime",
    version=version,
    author="Vinay Kamble",
    author_email="vinaykamble289@gmail.com",
    description="Fast ML inference runtime - 10-100x faster than joblib",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vinaykamble289/mle-runtime",
    packages=find_packages(),
    ext_modules=ext_modules,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ],
    },
    keywords="machine-learning inference ml ai runtime mle joblib",
)
