from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import versioneer


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __str__(self):
        import pybind11
        return pybind11.get_include()


# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    import os
    with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flagname])
    except setuptools.distutils.errors.CompileError:
        return False
    finally:
        try:
            os.remove(fname)
        except OSError:
            pass
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.

    The newer version is prefered over c++11 (when it is available).
    """
    flags = ['-std=c++17', '-std=c++14', '-std=c++11']

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    msg = 'Unsupported compiler -- at least C++11 support is needed!'
    raise RuntimeError(msg)


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {'msvc': ['/EHsc'], 'unix': [], }
    l_opts = {'msvc': [], 'unix': [], }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')

        for ext in self.extensions:
            ext.define_macros = [
                ('VERSION_INFO', '"{}"'.format(self.distribution.get_version()))]
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


with open("README.md", "r") as fh:
    long_description = fh.read()

xsum_modules = [Extension('xsum',
                          sorted(['xsum/xsum.cpp']),
                          include_dirs=[get_pybind_include(), ],
                          language='c++'
                          ), ]

setup(
    name='xsum',
    version=versioneer.get_version(),
    description='Fast Exact Summation Using Small and Large Superaccumulators',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/yafshar/xsum',
    author='Yaser Afshar',
    author_email='ya.afshar@gmail.com',
    license='LGPLv2',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)'
    ],
    setup_requires=['pybind11>=2.5.0'],
    keywords=['xsum'],
    packages=find_packages(),
    install_requires=['numpy'],
    cmdclass={'build_ext': BuildExt},
    ext_modules=xsum_modules,
    zip_safe=False,
)
