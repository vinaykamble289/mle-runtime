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

# C++ extension module - Skip for now, use bindings/python instead
ext_modules = []

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
    package_data={
        'mle_runtime': ['*.py'],
    },
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
        "sklearn": ["scikit-learn>=1.0.0", "joblib>=1.0.0"],
        "pytorch": ["torch>=1.10.0"],
        "tensorflow": ["tensorflow>=2.8.0"],
        "xgboost": ["xgboost>=1.5.0"],
        "lightgbm": ["lightgbm>=3.3.0"],
        "catboost": ["catboost>=1.0.0"],
        "all": [
            "scikit-learn>=1.0.0",
            "joblib>=1.0.0",
            "torch>=1.10.0",
            "tensorflow>=2.8.0",
            "xgboost>=1.5.0",
            "lightgbm>=3.3.0",
            "catboost>=1.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        'console_scripts': [
            'mle-export=mle_runtime.universal_exporter:main',
            'mle-export-sklearn=mle_runtime.sklearn_to_mle:main',
            'mle-export-pytorch=mle_runtime.pytorch_to_mle:main',
            'mle-export-tensorflow=mle_runtime.tensorflow_to_mle:main',
            'mle-export-xgboost=mle_runtime.xgboost_to_mle:main',
        ],
    },
    keywords="machine-learning inference ml ai runtime mle joblib",
)
