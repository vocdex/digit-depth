from setuptools import setup, find_packages

setup(
    name="digit-depth",
    version="0.1",
    description="Digit Depth Reconstruction",
    url="https://github.com/vocdex/digit-depth",
    author="Shukrullo Nazirjonov",
    author_email="nazirjonovsh2000@gmail.com",
    license="MIT",
    install_requires=['numpy', 'opencv-python', 'torch'],
    packages=find_packages(),
    zip_safe=False,
    extras_require={
     'testing': ['pytest'],
    },
)
