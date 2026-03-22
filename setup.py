from setuptools import setup, find_namespace_packages

setup(
    name="cli-anything-ollama",
    version="1.0.0",
    description="CLI-Anything harness for Ollama — run, manage, and script LLMs from the command line",
    author="cli-anything contributors",
    python_requires=">=3.10",
    packages=find_namespace_packages(include=["cli_anything.*"]),
    package_data={
        "cli_anything.ollama": ["skills/*.md"],
    },
    install_requires=[
        "click>=8.0",
        "requests>=2.28",
        "prompt_toolkit>=3.0",
    ],
    entry_points={
        "console_scripts": [
            "cli-anything-ollama=cli_anything.ollama.ollama_cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
