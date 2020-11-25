__description__ = \
"""
Software for processing phage display data for use with hops to learn peptide
binding rules.  Counts sequences in a fastq file (with quality control), clusters
similar sequences, determines enrichment, and determines the probability that a 
sequence arose from an enrichment versus random binding process.
"""

__version__ = "v0.1"

import sys 
if sys.version_info[0] < 3:
    sys.exit('Sorry, Python < 3.x is not supported')

from setuptools import setup, find_packages
import numpy

setup(name="hops_enrich",
      packages=find_packages(),
      version=__version__,
      description="software for processing phage display data for machine learning with hops",
      long_description=__description__,
      author='Michael J. Harms',
      author_email='harmsm@gmail.com',
      url='https://github.com/harmslab/hops_enrich',
      download_url="https://github.com/harmslab/fast_dbscan/archive/{}.tar.gz".format(__version__),
      install_requires=["numpy","fast_dbscan","hops"],
      package_data={},
      zip_safe=False,
      classifiers=['Programming Language :: Python'],
      entry_points = {
            'console_scripts': [
                  'hops_count = hops_enrich.count:main',
                  'hops_cluster = hops_enrich.cluster:main',
                  'hops_enrich = hops_enrich.enrich:main'
            ]
      })
