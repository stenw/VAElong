from setuptools import setup, find_packages

setup(
    name='vaelong',
    version='0.1.0',
    description='Variational Autoencoder for Longitudinal Measurements',
    author='',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'pyarrow>=14.0.0',
        'PyYAML>=6.0.0',
        'matplotlib>=3.7.0',
        'scikit-learn>=1.3.0',
        'statsmodels>=0.14.0',
        'nbformat>=5.9.0',
        'nbclient>=0.8.0',
        'nbconvert>=7.0.0',
    ],
    python_requires='>=3.8',
)
