from setuptools import setup, find_packages
pkgs = find_packages(exclude=["bin", "*.tests"])
print(pkgs)
setup(
    name="Textify",
    version="0.0.1",
    license="MIT",
    description="Deep Learning Models for Text Classification and Analysis using TensorFlow",
    author="Mohammed Jabreel",
    author_email="mhjabreel@gmail.com",
    url="http://mhjabreel.net",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    keywords="tensorflow text classifcation",
    install_requires=[
        "pyyaml",
    ],
    extras_require={
        "tensorflow": ["tensorflow>=1.4.0"],
        "tensorflow_gpu": ["tensorflow-gpu>=1.4.0"]
    },
    packages=find_packages(exclude=["bin", "*.tests", "demo"]),
    entry_points={
        "console_scripts": [
            "textify=textify.bin.main:main",
        ],
    }
)