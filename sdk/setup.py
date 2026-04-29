from setuptools import setup, find_packages

setup(
    name="deepsafe-sdk",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "Pillow>=9.0.0",
    ],
    entry_points={
        "console_scripts": [
            "deepsafe=deepsafe_sdk.server:cli",
        ],
    },
    python_requires=">=3.9",
)
