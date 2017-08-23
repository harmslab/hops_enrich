# pepenrich

Tools for processing phage display data to allow machine learning using peplearn

### Installation

#### Pip install (recommended)

Open a terminal and type:

```
pip3 install pepenrich
```

#### Manual installation
`pepenrich` requires python3.  It depends on `numpy`, `scipy`, `matplotlib`, and
`fast_dbscan`.  Once these are installed, open a terminal in a convenient
location on your computer and run:

```
git clone https://github.com/harmslab/pepenrich
cd pepenrich/
sudo python3 setup.py install
```

### Quick start

After installing, make a directory that has two `fastq.gz` files holding
Illumina reads for the experiment done in the presence and absence of
competitor.  In the example, these are called `alone.tar.gz` and 
`competitor.tar.gz`.  Then run:

```
pep_enrich alone.tar.gz competitor.tar.gz
```

This will calculate the enrichment of peptides in the competitor relative to 
the sample alone.  For a full list of options, type:

```
pep_enrich -h
```

### Scripts

This will install the following scripts in the console path:

`pep_count`: count peptides in an Illumina run
`pep_cluster`: cluster peptides by sequence
`pep_enrich`: calculate enrichment of peptides between runs
