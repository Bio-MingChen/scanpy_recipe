from setuptools import setup, find_packages

setup(
    name='scanpy_recipe',
    version='0.11',
    packages=find_packages(),
    description='shortcut tools for scRNA-seq data analysis based on scanpy',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='chenming',
    author_email='cm.bio@qq.com',
    url='https://github.com/Bio-MingChen/scanpy_recipe',
    install_requires=[
        'scanpy',
        'pandas',
        'numpy',
        'anndata',
        'scipy',
        'matplotlib',
        'celltypist',
        'scanorama',
    ],
)
