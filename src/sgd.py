
import pandas as pd


def extract_desc_val(data, key):
    """Extract the values in the description field of SGD"""
    def _extract_desc_val_row(orf_row, key):
        """Parse the description of a orf_row in sgd gff, extract `keys` and return 
        as a series. Can be used to return as a df when called with apply"""

        des_map = {}
        for entry in orf_row.desc.split(';'):
            k, val = tuple(entry.split('='))
            if k == key:
                return val

        return None

    # parse description column to extract relevant columns
    vals = data.apply(lambda row: _extract_desc_val_row(row, key=key),
        axis=1)

    return vals


def read_sgd_genes(filename='data/sgd_R64-1-1_20110208.gff'):
    """Read sgd orf/genes file as tsv file from gff file with fasta data removed."""

    data = pd.read_csv(filename, sep='\t', skiprows=19, 
                              names=["chr", "source", "cat", "start", "stop", ".", 
                              "strand", "", "desc"])
    data = data[data.columns[[0, 2, 3, 4, 6, 8]]]
    data.columns = ["chr", "cat", "start", "stop", "strand", "desc"]
    data = data[data['cat'] == 'gene']

    gene_names = extract_desc_val(data, 'gene') 
    systematic_name = extract_desc_val(data, 'ID') 
        
    data['gene'] = gene_names
    data['orf_name'] = systematic_name

    data = data[['orf_name', 'gene', "chr", "cat", "start", "stop", "strand", 'desc']].set_index('orf_name')
    
    return data
