from setuptools import setup, find_packages

with open("requirements.txt",) as f:
    requirements = f.read().splitlines()

setup(
    name="Anime-Hybrid-Recommender-System",
    version="0.1",
    author="Ali-RVN",
    packages=find_packages(),
    install_requires=requirements,
)