# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from io import open
import versioneer

with open('requirements.txt', encoding='utf-8') as requirements:
    requires = [l.strip() for l in requirements]

with open('README.md', encoding='utf-8') as readme_f:
    readme = readme_f.read()

author = 'Fidel Ramírez, Gina Renschler, Gautier Richard'

setup(
    name='HiCAssembler',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Hi-C guided genome assembly',
    long_description=readme,
    url='https://github.com/maxplanck-ie/HiCAssembler',
    author=author,
    author_email='fidel.ramirez@gmail.com',
    license='BSD',
    python_requires='>=2.7',
    install_requires=requires,
    extras_require=dict(
        test=['pytest'],
    ),
    packages=find_packages(exclude=['tests']),
    scripts=['bin/assemble', 'bin/plotScaffoldInteractive', 'bin/plotScaffoldsHiC'],
    package_data={'': '*.txt'},
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Framework :: Jupyter',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)
