from setuptools import setup, find_packages

setup(
    name='amworkflow',
    version='1.0',
    include_package_data=True,
    packages=find_packages(),
    url='',
    license='MIT',
    author='Yuxiang He',
    author_email='yuxiang.he@bam.de',
    description='',
    install_requires=[
        "gmsh>=4.11.1",
        "networkx",
        "sphinx",
        "sqlalchemy",
        "ruamel-yaml",
        "psutil",
        "pyqt5"
    ]
)
