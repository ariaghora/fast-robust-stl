from setuptools import setup

setup(
    name='frstl',
    version='0.1.0',
    description=('Unofficial implementation of "Fast RobustSTL: Efficient and Robust Seasonal-Trend Decomposition for Time Series with Complex Patterns"'),
    author='Aria Ghora Prabono',
    author_email='hello@ghora.net',
    url='https://github.com/ariaghora/torch_kernel',
    license='MIT',
    packages=['frstl'],
    include_package_data=True,
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.8'],
    )