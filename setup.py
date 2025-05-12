from setuptools import setup, find_packages

setup(
    name="machine_translation ge to en",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "datasets>=1.8.0",
        "spacy>=3.0.0",
        "nltk>=3.5",
    ],
    author="Sunil Reddy",
    author_email="challamalla5sunil@gmail.com",
    description="A German-to-English machine translation package using PyTorch",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/machine_translation",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)