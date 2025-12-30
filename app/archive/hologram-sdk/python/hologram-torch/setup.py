#!/usr/bin/env python3
"""
Hologram PyTorch Extension - Native torch.device('hologram') support

This package provides native PyTorch device backend integration for Hologram,
enabling seamless use of Hologram's canonical compilation engine within PyTorch.
"""

import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        import subprocess
        import torch
        import sysconfig

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Get PyTorch CMake prefix path
        torch_cmake_prefix = torch.utils.cmake_prefix_path

        # Get Python extension suffix
        ext_suffix = sysconfig.get_config_var('EXT_SUFFIX') or '.so'

        # CMake configuration arguments
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCMAKE_PREFIX_PATH={torch_cmake_prefix}',
            f'-DPython_EXTENSION_SUFFIX={ext_suffix}',
        ]

        # Build configuration
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        # Platform-specific arguments
        if sys.platform.startswith('darwin'):
            # macOS: Set deployment target
            cmake_args += ['-DCMAKE_OSX_DEPLOYMENT_TARGET=10.15']

        # Build
        cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']
        build_args += ['--', '-j4']

        # Create build directory
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Run CMake
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


setup(
    name='hologram-torch',
    version='0.1.0',
    author='Hologram Contributors',
    description='Native PyTorch backend for Hologram',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    ext_modules=[CMakeExtension('hologram_torch._hologram_torch')],
    cmdclass={'build_ext': CMakeBuild},
    packages=['hologram_torch'],
    package_dir={'hologram_torch': 'python/hologram_torch'},
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
    ],
)
