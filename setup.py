from setuptools import setup
import os
path_to_am = os.getcwd()
setup(
    name='amworkflow',
    version='1.0',
    packages=['amworkflow.src', 'amworkflow.tests'],
    install_requires=[f"OCCUtils-0.1.dev0 @ file://localhost/{path_to_am}/amworkflow/dependencies/OCCUtils-0.1.dev0-py3-none-any.whl"],
    package_data={
        # Include the dependencies along with your package
        'my_package': ['amworkflow.dependencies/*']
    },
    url='',
    license='MIT',
    author='Yuxiang He',
    author_email='yuxiang.he@bam.de',
    description=''
)
