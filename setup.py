from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

import os
import shutil

class my_build_ext(build_ext):
    def build_extension(self, ext):
        """Simply copy the first source file to the destination

        I did my best at doing something sensical after looking at the
        distutils sources.  I don't handle inplace or anything like
        that.

        """
        fullname = self.get_ext_fullname(ext.name)
        ext_filename = os.path.join(self.build_lib,
                                    self.get_ext_filename(fullname))
        shutil.copy(ext.sources[0], ext_filename)

setup(
    cmdclass = {'build_ext':my_build_ext},
    name = "ViVid",
    version = "0.0",
    packages = ['vivid'],
    #ext_modules = [Extension('vivid._vivid', ['vivid/_vivid.so'])]
    #package_dir = {'':'vivid'}
)
