"""Setup configuration for Chisel."""

from setuptools import setup, find_packages

setup(
    name="chisel-cli",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "typer>=0.16.0",
        "pydo>=0.4.0",
        "toml>=0.10.2",
        "rich>=14.0.0",
    ],
    entry_points={
        "console_scripts": [
            "chisel=chisel.main:main",
        ],
    },
    python_requires=">=3.8",
    author="Chisel Contributors",
    description="A CLI tool for managing DigitalOcean GPU droplets for HIP kernel development",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)