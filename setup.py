from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='simplemaml',
    version='1.01',
    description='A generic Python function that implements a simple version of the "Model-Agnostic Meta-Learning (MAML) Algorithm for Fast Adaptation of Deep Networks" as designed by Chelsea Finn et al. 2017',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Anas Neumann',
    author_email='anas.neumann.1@ulaval.ca',
    url='https://github.com/AnasNeumann/simplemaml',
    packages=find_packages(),
    py_modules=['maml'],
    install_requires=required,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)
