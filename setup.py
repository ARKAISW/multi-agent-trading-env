from setuptools import setup, find_packages

setup(
    name="mate-env",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "numpy",
        "pandas",
        "matplotlib",
        "yfinance",
        "fastapi",
        "uvicorn",
        "openai",
    ],
    author="Arka Sarkar",
    description="Multi-Agent Trading Environment for the OpenEnv Hackathon",
)
