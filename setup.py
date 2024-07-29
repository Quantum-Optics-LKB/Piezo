"""Init module for the Piezo package."""

from setuptools import setup

setup(
    name="Piezo",
    version="0.0.1",
    description="Python interface for Thorlabs piezo devices",
    url="https://github.com/Quantum-Optics-LKB/Piezo",
    author="Tangui Aladjidi, Lucien Belzane",
    author_email="tangui.aladjidi@lkb.upmc.fr",
    license="GPLv3",
    license_files=["LICENSE"],
    packages=["Piezo"],
    install_requires=[
        "numpy",
        "pythonnet",
    ],
    extra_requires=["pylablib", "thorlabs_apt_device"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
