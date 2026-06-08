from setuptools import setup, find_packages
setup(
    name="cpu-distributed",
    version="0.1.0",
    author="Ishmael Affum Kwakye",
    description="Distributed CPU training via evolution strategies",
    url="https://github.com/IshCPU-VideoGenLab/cpu-distributed",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=["torch>=2.0.0", "numpy>=1.24.0"],
)
