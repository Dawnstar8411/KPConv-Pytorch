from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

SOURCES = ["knn.pyx",
           "knn_.cxx"]

ext_modules = Extension(
    "nearest_neighbors",
    sources=SOURCES,  # 源文件
    include_dirs=["./", numpy.get_include()],
    language="c++",
    extra_compile_args=["-std=c++11", "-fopenmp", ],
    extra_link_args=["-std=c++11", '-fopenmp'],
)

setup(
    name="KNN NanoFLANN",
    ext_modules=[ext_modules],
    cmdclass={'build_ext': build_ext},
)
