from distutils.core import setup, Extension

# the c++ extension module
extension_mod = Extension("cec", ["pycec.cpp", "cec17.cpp"])

setup(name = "cec", ext_modules=[extension_mod])
