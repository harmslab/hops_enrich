
# Calculate enrichment of peptides between these files, appending "example" to
# front of all output files.  Do clustering on input sequences and set the 
# minimum number of counts to include a peptide to 6
hops_enrich alone.fastq.gz competitor.fastq.gz -b example -c -m 6
