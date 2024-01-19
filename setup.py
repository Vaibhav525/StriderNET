from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.13'
DESCRIPTION = 'Graph reinforcement learning based optimizer for atomic systems'
LONG_DESCRIPTION = 'A package to train graph reinforcement learning optimizer model for atomic structures on rough energy landscape'

# Setting up
setup(
    name="StriderNet",
    version=VERSION,
    author="VaibhavBihani",
    author_email="<vaibhav.bihani525@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['torch', 'torch-geometric'],
    keywords=['reinforcement learning', 'machine learning', 'optimization', 'rough landscapes', 'atomic'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)