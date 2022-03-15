from pkg_resources import parse_requirements
from setuptools import setup

long_description = open('README.md').read()
install_requires = [str(r) for r in parse_requirements(open('requirements.txt'))]

setup(
    name='vmodel',
    version='0.0.1',
    author='Fabian Schilling',
    email='fabian@schilli.ng',
    packages=['vmodel'],
    license='LICENSE',
    scripts=['scripts/vmodel', 'scripts/vmetric', 'scripts/vmerge'],
    description='Visual model',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=install_requires,
)
