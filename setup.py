from setuptools import setup

__version__ = '1.2.0'


setup(
    name='stanhelper',
    packages=['stanhelper'],
    version=__version__,
    description='Functions that help interface with cmdStan.',
    author='Alp Kucukelbir',
    author_email='alp@cs.columbia.edu',
    url='https://github.com/akucukelbir/stanhelper',
    download_url=('https://github.com/akucukelbir/stanhelper/releases/' +
                  __version__),
    install_requires=['pandas'],
    python_requires='>=3.5',
    license='MIT',
    keywords=[],
    classifiers=[],
)
