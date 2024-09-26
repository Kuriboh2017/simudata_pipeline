from setuptools import setup, find_packages


setup(
    name='fisheye',
    version='1.0',
    description='A useful module',
    license="MIT",
    author='Man fisheye',
    author_email='fisheyemail@fisheye.example',
    url="http://www.fisheyepackage.example/",
    packages=find_packages(include=['fisheye', 'fisheye.*']),
    # external packages as dependencies
    install_requires=['wheel', 'scipy'],

)
