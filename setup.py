"""
Setup script for agents
"""

from setuptools import setup, find_packages

with open("agents/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ollama-agents",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A lightweight, modular framework for building agents with Ollama models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ollama-agents",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "aiohttp>=3.8.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "distributed": [
            "sollol>=0.1.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.20.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            # Add CLI tools here if needed
        ],
    },
)
