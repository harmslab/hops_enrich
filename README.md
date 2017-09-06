# hops_enrich

Tools for processing phage display data to allow machine learning using hops.

### Installation

#### Pip install (recommended)

Open a terminal and type:

```
pip3 install hops_enrich
```

#### Manual installation
`hops_enrich` requires python3.  It depends on `numpy`, `scipy`, `matplotlib`, and
`fast_dbscan`.  Once these are installed, open a terminal in a convenient
location on your computer and run:

```
git clone https://github.com/harmslab/hops_enrich
cd hops_enrich/
sudo python3 setup.py install
```

### Quick start

After installing, make a directory that has two `fastq.gz` files holding
Illumina reads for the experiment done in the presence and absence of
competitor.  In the example, these are called `alone.tar.gz` and 
`competitor.tar.gz`.  Then run:

```
hops_enrich alone.tar.gz competitor.tar.gz
```

This will calculate the enrichment of peptides in the competitor relative to 
the sample alone.  For a full list of options, type:

```
hops_enrich -h
```

### Scripts

This will install the following scripts in the console path:

`hops_count`: count peptides in an Illumina run
`hops_cluster`: cluster peptides by sequence
`hops_enrich`: calculate enrichment of peptides between runs
