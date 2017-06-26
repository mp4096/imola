"""Setup file for Imola."""

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name="imola",
    version="0.1",
    description="Simulation and benchmarking environment "
    "for road lane estimators.",
    license="BSD",
    author="Mikhail Pak <mikhail.pak@tum.de>",
    packages=["imola"],
    install_requires=["numpy", "scipy", "pyyaml"]
    )
