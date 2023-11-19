from setuptools import setup

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='simplemaml',
    version='1.2.10',
    description='A generic Python and TensorFlow function that implements a simple version of the "Model-Agnostic Meta-Learning (MAML) Algorithm for Fast Adaptation of Deep Networks" as designed by Chelsea Finn et al. 2017',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Anas Neumann',
    author_email='anas.neumann.1@ulaval.ca',
    url='https://github.com/AnasNeumann/simplemaml',
    py_modules=['simplemaml'],
    install_requires=["numpy", "tensorflow"],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)