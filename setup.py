from setuptools import setup, find_packages

setup(
    name='stable-diffusion',
    version='0.1',
    description='',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'pytorch-lightning',
        'numpy',
        'tqdm',
        'einops'
    ],
)
