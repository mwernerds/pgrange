from distutils.core import setup, Extension

module1 = Extension('pgrange',
                    sources = ['pgrange.cpp'],
                    extra_compile_args=['-std=c++11','-I/usr/lib/postgresql/9.5/bin','-I/usr/include/postgresql'],
		    		        libraries=['pq'])

# example for extra_compile_args=['-std=c++11','-I/usr/lib/postgresql/9.5/bin','-I/usr/include/postgresql']

setup (name = 'pgrange',
       version = '1.1',
       description = 'This is a tool for fast range queries for Deep Learning from LiDAR cells',
       ext_modules = [module1])
