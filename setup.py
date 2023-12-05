from setuptools import setup, find_packages

setup(
    name='segmentation_package',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'numpy',
        'pandas',
        # Add any other dependencies required by your functions
    ],
)
