from setuptools import setup, find_packages

setup(
    name='ImageSegmentation',
    version='1.0',
    packages=find_packages(),
    description='Image segmentation tools using watershed algorithm',
    author='Hajar',
    author_email='moradmand90@gmail.com',
    install_requires=[
        'opencv-python',
        'numpy',
        'pandas',
        # Add any other dependencies required by your functions
    ],
    license='MIT'
)
