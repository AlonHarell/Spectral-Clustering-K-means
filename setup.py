from setuptools import setup, Extension
module = Extension("spkmeans",sources=['spkmeans.c','spkmeansmodule.c'])
setup(name='spkmeans',ext_modules=[module])