
from src.utils import _fromRoman
import pandas as pd
import numpy as np
from src.utils import print_fl


def read_orfs_data(filename=None, index_name='orf_name', times=[0, 7.5, 15, 30, 60, 120]):
    """
    Load ORFs data set with time columns converting time columns from string
    to integers if necessary. Assumes file is a csv with columns. Assumes there
    is an orf_name column that can be set as the index.
    """

    data = pd.read_csv(filename).set_index(index_name)
    
    non_time_cols = list(data.columns)
    int_time_cols = []
    for time in times:

        # check if time column exists
        str_time_col = "%.1f" % time
        if not str_time_col in data.columns:
            str_time_col = str(int(time))
            if not str_time_col in data.columns:
                continue

        # rename column to numeric
        data = data.rename(columns={str_time_col:time})

    return data

def load_danpos_nucs(time):
    danpos_nucs = pd.read_csv(f'output/cac1_pulse_chase/danpos_nucs_{time}.csv')
    return danpos_nucs


def read_gene_origin_directionality():
    orc_directions = pd.read_csv('data/sacCer3_ucsc_geneTable_COMPLETE.csv')
    orc_directions.loc[orc_directions.directionality == '#NAME?', 'directionality'] = float('inf')
    orc_directions.directionality = orc_directions.directionality.astype('float')
    orc_directions = orc_directions[orc_directions.directionality < float('inf')].drop('Unnamed: 0',
        axis=1).set_index('name')
    orc_directions['chr'] = orc_directions['chr'].str.replace('chr', '').map(_fromRoman).astype(int)
    orc_directions = orc_directions.sort_values(['chr', 'start'])
    return orc_directions


def read_robocop_nucs():
    nucs = pd.read_csv('data/nuc_map_categories.csv', sep='\t')
    nucs['chr'] = nucs['chr'].str.replace('chr', '').map(_fromRoman).astype(int)
    
    return nucs


def read_arss():
    
    data = read_sgd()
    arss = data[data['cat'] == 'ARS'].copy()
    arss['length'] = arss['stop']-arss['start']
    arss['name'] = extract_desc_val(arss, 'ID')
    arss = arss[['name', 'chr', 'cat', 'start', 'stop', 'length']]
    
    return arss


def read_centromeres():

    data = read_sgd()

    cens = data[data['cat'] == 'centromere'].copy()
    cens = cens[['chr', 'cat', 'start', 'stop', 'strand']]
    return cens


def read_sgd():
    """Read sgd orf/genes file as tsv file from gff file with fasta data removed."""

    filename = 'data/saccharomyces_cerevisiae_R64-1-1_20110208_no_fasta.gff'
    data = pd.read_csv(filename, sep='\t', skiprows=19, 
                              names=["chr", "source", "cat", "start", "stop", ".", 
                              "strand", "", "desc"])
    data = data[data.columns[[0, 2, 3, 4, 6, 8]]]
    data.columns = ["chr", "cat", "start", "stop", "strand", "desc"]
    data.chr = parse_roman_chr(data.chr)

    return data


def read_chromosome_span(chrom):
    data = read_sgd()
    data = data[(data['chr'] == chrom) & (data['cat'] == 'chromosome')].reset_index(drop=True).loc[0]
    chrom_span = tuple(data[['start', 'stop']])
    return chrom_span


def read_centromere(chrom):
    data = read_sgd()
    data = data[(data['chr'] == chrom) & (data['cat'] == 'centromere')].reset_index(drop=True).loc[0]
    centromere = (data['start'] + data['stop'])//2
    return centromere


def read_macisaac_sites(tf=None):
    sites = pd.read_csv('data/p005_c2.sacCer3.gff.txt', sep='\t',
               names=range(9))
    sites = sites[sites.columns[[0, 3, 4, 6, 8]]].copy()
    sites.columns = ['chr','start','stop','strand','TF']

    sites.chr = parse_roman_chr(sites.chr)
    sites.TF = sites.TF.str.replace(';', '').str.replace('Site ', '')
    if tf is not None:
        sites = sites[sites.TF.str.lower() == tf.lower()].copy()
    sites['mid'] = ((sites['stop'] + sites['start'])/2).astype(int)

    return sites


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


def orfs_for_go_term(go_term):
    orfs_data = read_sgd_orfs()
    orfs = orfs_data[[go_term in o for o in orfs_data.ontology.str.split(',')]].index.values
    return orfs


def read_sgd_orfs():

    from pandas import Series

    data = read_sgd()
    orfs = data[data['cat'] == 'gene'].copy()
    orfs['orf_name'] = None
    orfs['orf_class'] = None

    # chromosomal orfs
    orfs = orfs[orfs.chr > 0]

    orf_names = extract_desc_val(orfs, 'ID') 
    orf_classes = extract_desc_val(orfs, 'orf_classification')
    names = extract_desc_val(orfs, 'gene') 
    ontology = extract_desc_val(orfs, 'Ontology_term') 

    orfs['orf_name'] = orf_names
    orfs['orf_class'] = orf_classes

    # set name if it exists, or orf_name by default
    orfs['name'] = names
    orfs.loc[orfs.name.isna(), 'name'] = orfs[orfs.name.isna()]['orf_name']
    orfs['ontology'] = ontology

    orfs['length'] = orfs['stop'] - orfs['start'] + 1
    orfs = orfs.set_index('orf_name')[[
        'name','chr','start','stop','length','strand','orf_class', 'ontology']]

    return orfs


def read_sgd_orf_introns():
    """Read sgd orf/genes file as tsv file from gff file with fasta data removed."""

    genes = read_sgd_orfs()
    data = read_sgd()

    keep_classes = ['intron', 'five_prime_UTR_intron', 'CDS']
    introns_CDSs = data[data.cat.isin(keep_classes)].copy()

    # rename 5' class to intron
    introns_CDSs.loc[(introns_CDSs['cat'] == 'five_prime_UTR_intron'), 'cat'] = 'intron'
    introns_CDSs['parent'] = extract_desc_val(introns_CDSs, 'Parent') 

    # relevant genes
    introns_CDSs = introns_CDSs[introns_CDSs.parent.isin(genes.index.values)]

    return introns_CDSs[['cat', 'start', 'stop', 'parent']].copy().reset_index(drop=True)


def read_rossi_tf(tf):

    bed_dir = 'output/rossi_2021_sites_bed'
    tf = tf.title()
    
    file = f'{tf}.multi_{tf}.filtered.bed'
    bed_file = f"{bed_dir}/{file}"

    tf_name = file.split('.')[0]
    
    sites = pd.read_csv(bed_file, sep='\t', names=['chr', 'start', 'end', 'cat', 'score'])
    print(f"{len(sites)} {tf_name} sites")

    sites['mid'] = (sites['start'] + sites['end'])//2
    sites['chr'] = sites['chr'].str.replace('chr', '').map(_fromRoman).astype(int)

    return sites


def parse_roman_chr(series):
    return series.str.replace('chr', '').apply(_fromRoman)


def all_orfs_TSS_PAS():
    all_orfs = read_sgd_orfs()
    all_orfs = all_orfs.join(read_park_TSS_PAS()[['TSS', 'PAS']])
    return all_orfs


def read_park_TSS_PAS():

    TSS_filename = 'data/Park_2014_TSS_V64.gff'
    PAS_filename = 'data/Park_2014_PAS_V64.gff'

    def _read_park_gff(filename, key):
        park_data = pd.read_csv(filename, sep='\t', skiprows=3)
        columns = park_data.columns[[0, 2, 3]] # relevant columns
        park_data = park_data[columns].copy()

        # cleanup
        park_data.columns = ['chr', 'orf_name', key]
        park_data.chr = parse_roman_chr(park_data.chr)
        park_data.orf_name = park_data.orf_name.str.replace('_%s' % key, '')
        park_data[key] = park_data[key].astype(int)

        return park_data.set_index('orf_name')

    TSS = _read_park_gff(TSS_filename, 'TSS')

    # manually annotated/adjusted TSSs for vignettes
    TSS.loc['YBR072W', 'TSS'] = 381753
    TSS.loc['YDR253C', 'TSS'] = 964767
    TSS.loc['YBR294W', 'TSS'] = 789000
    TSS.loc['YLR092W', 'TSS'] = 323500
    TSS.loc['YOL164W', 'TSS'] = 6000

    PAS = _read_park_gff(PAS_filename, 'PAS')
    data = TSS[['TSS']].join(PAS[['PAS']])

    data['manually_curated'] = False
    data.loc['YBR072W', 'manually_curated'] = True
    data.loc['YDR253C', 'manually_curated'] = True
    data.loc['YBR294W', 'manually_curated'] = True
    data.loc['YLR092W', 'manually_curated'] = True
    data.loc['YOL164W', 'manually_curated'] = True

    return data
