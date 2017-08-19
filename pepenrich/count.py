#!/usr/bin/env python3
__description__ = \
"""
Grab the number of times each pepetide was seen in a fastq file, performing 
quality control on each read.  Quality control is:
    1. Is the sequence translatable in-frame, without stops or nonsensical 
       codons?
    2. Is the average PHRED score above a cutoff (15 by default)
    3. Is the flanking phage region correct to within one base across the whole
       sequence?
"""
__author__ = "Michael J. Harms"
__date__ = "2015-04-28"

import numpy as np
import os, gzip, pickle, re, sys, argparse

class FastqSeqCounter:
    """
    Class for converting a set of fastq nucleotide sequences into a dictionary 
    mapping peptide sequences to counts.  Does quality control as part of the
    processing.
    """

    def __init__(self,bad_pattern="[*X]",seq_length=12,phred_cutoff=15):

        self.phred_cutoff = phred_cutoff

        # These are all amino acid sequences accessible as single point mutants
        # from the normal phage sequence GGGSAE
        self._ALLOWED_PHAGE_SEQ = ["GGSSAE","GGGSAE","AGGSAE","GGGSDE","GGASAE",
                                   "GGGSPE","DGGSAE","GGGSSE","VGGSAE","GGGSTE",
                                   "GGRSAE","GEGSAE","GGGPAE","GGGSAV","GGGSVE",
                                   "GGGLAE","GGGAAE","RGGSAE","GGGSAG","GGGTAE",
                                   "CGGSAE","GGGSAQ","GGG*AE","GGGSGE","GVGSAE",
                                   "G*GSAE","GAGSAE","GGGSA*","SGGSAE","GGVSAE",
                                   "GRGSAE","GGGSAD","GGGSAA","GGGWAE","GGDSAE",
                                   "GGGSAK","GGCSAE"]
        self._GENCODE = {
            'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
            'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
            'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
            'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
            'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
            'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
            'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
            'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
            'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
            'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
            'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
            'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
            'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
            'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
            'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
            'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W'}

        self._phage_part_size = len(self._ALLOWED_PHAGE_SEQ[0])
        self._bad_pattern = re.compile(bad_pattern)
        self._seq_length = seq_length

        # Dictionary mapping ascii value to Q-score
        self._Q_dict = {}
        for i in range(33,76):
            self._Q_dict[chr(i)] = i-33

    def _translate(self,sequence):
        """
        Translate a nucleotide sequence into a protein sequence.  If there is a
        problem, write an "X" into the sequence.
        """

        try:
            return "".join([self._GENCODE[sequence[3*i:3*i+3]]
                            for i in range(len(sequence)//3)])
        except KeyError:
            out = []
            for i in range(len(sequence)//3):
                try:
                    out.append(self._GENCODE[sequence[3*i:3*i+3]])
                except KeyError:
                    out.append("X")
            return "".join(out)

    def _qualityCheck(self,sequence,phred):
        """ 
        Make sure that the C-terminus of the peptide is what we think it should
        be and that there aren't any "_bad_patterns" in the protein sequence. 
        (As of this writing, _bad_pattern will catch "X" and premature stop 
        codons). 
        """
       
        # Make sure Phred scores are good 
        if np.mean([self._Q_dict[p] for p in phred]) < self.phred_cutoff:
            return False
       
        # Make sure it looks like a phage display read 
        phage_part = sequence[self._seq_length:(self._seq_length+self._phage_part_size)]
        real_part = sequence[:self._seq_length]

        if phage_part in self._ALLOWED_PHAGE_SEQ:
            if not self._bad_pattern.search(real_part):
                return True

        return False
   
    def processFastqFile(self,fastq_file):
        """
        Create a set of good and bad pattern dicts for a given fastq file.
        """

        good_count_dict = {}
        bad_count_dict = {}

        line_counter = 0
        get_sequence = False
        get_phred = False
        with gzip.open(fastq_file,'r+') as f:
       
            for l in f:
            
                l_ascii = l.decode("ascii")

                if line_counter == 0:
                    line_counter += 1
                    continue
                elif line_counter == 1:
                    sequence = l_ascii.strip()
                    line_counter += 1
                    continue
                elif line_counter == 2:
                    line_counter += 1
                    continue
                elif line_counter == 3:
                    phred = l_ascii.strip()
                    line_counter = 0
                else:
                    err = "could not parse file\n"
                    raise ValueError(err)

                # Translate the sequence
                sequence = self._translate(sequence)

                # Record it in either the good or bad dict, depending on its
                # quality score
                if self._qualityCheck(sequence,phred):

                    key = sequence[0:self._seq_length]
                    try:
                        good_count_dict[key] += 1
                    except KeyError:
                        good_count_dict[key] = 1
                else:
                    try:
                        bad_count_dict[sequence] += 1
                    except KeyError:
                        bad_count_dict[sequence] = 1

        return good_count_dict, bad_count_dict

    @property
    def bad_pattern(self):
        """
        Return regular expression we use to look for "badness"
        """
        return self._bad_pattern.pattern

    @property
    def seq_length(self):
        """
        Return expected length of peptide sequences.
        """
        return self._seq_length


def fastq_to_count(fastq_filename,phred=15,out_file=None):
    """
    Process fastq file
    """

    # Create object to count the fastq sequences
    p = FastqSeqCounter(phred_cutoff=phred)

    # Count
    good_counts, bad_counts = p.processFastqFile(fastq_filename)

    if out_file is not None:
        f = open(out_file,'w')
        f.write("# phred cutoff: {}\n".format(phred))

        seqs = list(good_counts.keys())
        seqs.sort()
        for s in seqs:
            f.write("{} {}\n".format(s,good_counts[s])) 
        f.close()

    return good_counts

def read_counts(count_file):
    """
    Read a file containing sequences vs. counts.  Returns a dictionary of 
    frequencies keyed to sequences.  
    """

    f = open(count_file,'r')
    lines = f.readlines()
    f.close()

    count_dict = {}
    for l in lines:
        if l.strip() == "" or l[0] == "#":
            continue

        col = l.split()
        seq = col[0].strip()
        counts = int(col[1])

        count_dict[seq] = counts

    return count_dict

def main(argv=None):
    """
    Run the script on a fastq file.
    """

    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description=__description__)
        
    # Positionals
    parser.add_argument("fastq_file",help="gzipped fastq file from which peptides will be extracted and counted")

    # Options 
    parser.add_argument("-p","--phred",help="phred cutoff",action="store",type=int,default=15)
    parser.add_argument("-o","--out",help="output file",action="store",type=str,default=None)

    args = parser.parse_args(argv)

    if args.out_file is None:
        out_file = "{}.counts".format(args.fastq_file)
    else:
        out_file = args.out_file

    out_dict = fastq_to_count(fastq_filename=args.fastq_file,
                              phred=args.phred,
                              out_file=out_file)
    
if __name__ == "__main__":
    main() 
