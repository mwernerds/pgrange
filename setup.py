from distutils.core import setup, Extension

module1 = Extension('pgrange',
                    sources = ['pgrange.cpp'],
                    extra_compile_args=['-std=c++11','-I/usr/include/postgresql'],
                    libraries=['pq'])



setup (name = 'pgrange',
       version = '1.0',
       description = 'This is a tool for fast range queries for Deep Learning from LiDAR cells',
       ext_modules = [module1])
