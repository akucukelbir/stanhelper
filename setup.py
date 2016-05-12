from setuptools import setup

setup(
  name = 'stanhelper',
  packages = ['stanhelper'], 
  version = '0.7',
  description = 'Functions that help interface with cmdStan.',
  author = 'Alp Kucukelbir',
  author_email = 'alp@cs.columbia.edu',
  url = 'https://github.com/akucukelbir/stanhelper',
  download_url = 'https://github.com/akucukelbir/stanhelper/releases/0.7',
  install_requires=['numpy>=1.7', 'pandas>=0.18.0'],
  license='GPLv3',
  keywords = [],
  classifiers = [],
)
