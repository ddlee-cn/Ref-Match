# pybind11 template taken from: https://github.com/edmBernard/pybind11_opencv_numpy
# Original author of VFC: zhaoji84@gmail.com

import fnmatch
import os
from os.path import dirname, exists, join
from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import sys
import setuptools


__version__ = '0.0.1'

#
# pybind-specific compilation stuff
#

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.
    The c++14 is preferred over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-s') # strip
            opts.append('-g0') # remove debug symbols
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


ext_modules = [
    Extension(
        'pyvfc',
        [
            'pyvfc/pyvfc.cpp',
            'pyvfc/vfc.cpp',
        ],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
        ],
        libraries=['opencv_core', 'opencv_highgui'],
        language='c++',
    ),
]


setup(
    name='pyvfc',
    version=__version__,
    author='Peter Crampton',
    author_email='cramppet@protonmail.com',
    url='https://github.com/cramppet/pyvfc',
    description='Vector field consensus (VFC) Python wrapper',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.4', 'opencv-python==3.4.9.31'],
    setup_requires=['pybind11>=2.4', 'opencv-python==3.4.9.31'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
)