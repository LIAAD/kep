from setuptools import setup, find_packages
import os

requirementPath = 'requirements.txt'

if os.path.isfile(requirementPath):
    with open('requirements.txt') as f:
        requires = f.readlines()

install_requires = [item.strip() for item in requires]


setup(name="kep",
    version="0.1",
    description="Keyphrases Extraction Package",
    author='Ricardo Campos',
    author_email='ricardo.campos@ipt.pt',
    packages=find_packages(),
    include_package_data=True,
    install_requires=install_requires
)