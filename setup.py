__description__ = \
"""
Software for processing phage display data for use with peplearn to learn peptide
binding rules.  Counts sequences in a fastq file (with quality control), clusters
similar sequences, determines enrichment, and determines the probability that a 
sequence arose from an enrichment versus random binding process.
"""

__version__ = 0.1

import sys 
if sys.version_info[0] < 3:
    sys.exit('Sorry, Python < 3.x is not supported')

from setuptools import setup, find_packages
import numpy

setup(name="pepenrich",
      packages=find_packages(),
      version=__version__,
      description="software for processing phage display data for machine learning with peplearn",
      long_description=__description__,
      author='Michael J. Harms',
      author_email='harmsm@gmail.com',
      url='https://github.com/harmslab/pepenrich',
      download_url="https://github.com/harmslab/fast_dbscan/archive/{}.tar.gz".format(__version__),
      install_requires=["numpy","fast_dbscan","peplearn"],
      package_data={},
      zip_safe=False,
      classifiers=['Programming Language :: Python'],
      entry_points = {
            'console_scripts': [
                  'pep_count = pepenrich.count:main',
                  'pep_cluster = pepenrich.cluster:main',
                  'pep_enrich = pepenrich.enrich:main'
            ]
      })
