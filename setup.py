from setuptools import setup, find_packages

setup(
    name='scanpy_recipe',
    version='0.1',
    packages=find_packages(),
    description='shortcut tools for scRNA-seq data analysis based on scanpy',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='cm.bio@qq.com',
    url='https://github.com/yourusername/your_package_name',
    install_requires=[
        'some_package>=1.0',
    ],
)
