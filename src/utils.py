
import subprocess
import sys
import os

def print_fl(val='', end='\n', log=True):

    contents = str(val) + end

    sys.stdout.write(contents)
    sys.stdout.flush()


def mkdirs_safe(directories, log=True):
    for directory in directories:
        mkdir_safe(directory, log=log)


def mkdir_safe(directory, log=True):
    if log: print_fl("Creating directory: %s..." % directory, end='')

    if not os.path.exists(directory):
        os.makedirs(directory)
    elif log:
        print_fl("Directory exists. Skipping.", end='')

    if log: print_fl()


def _toRoman(number):
    try:
        return {1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V', 6: 'VI', 7: 'VII',
         8: 'VIII', 9: 'IX', 10: 'X', 11: 'XI', 12: 'XII', 13: 'XIII', 
         14: 'XIV', 15: 'XV', 16: 'XVI'}[number]
    except KeyError:
        return -1
    

def _fromRoman(roman):
    try:
        return {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, 
            "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10, 
            "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, 
            "XV": 15, "XVI": 16}[roman]
    except KeyError:
        return -1
    
def read_yeast_genome():
    """
    Read yeast genome data for looking up sequences at various locations in genome.
    """

    input_path = "/Users/trung/Documents/projects/heavy-metal/data/fimo/S288C_reference_sequence_R64-1-1_20110203.fsa"
    from Bio import SeqIO

    NUM_CHROM = 16

    sequences = {}
    
    with open(input_path, 'r') as input_file:
        fasta_sequences = SeqIO.parse(input_file,'fasta')

        for fasta in fasta_sequences:
            chrom = fasta.description.split(' ')[-1].replace('[chromosome=', '').replace(']', '')
            chrom = _fromRoman(chrom)

            name, sequence = fasta.id, fasta.seq
            if chrom > 0:
                sequences[chrom] = sequence

    return sequences


def run_cmd(bashCommand, stdout_file=None):
    if stdout_file is not None:
        with open(stdout_file, 'w') as output:
            process = subprocess.Popen(bashCommand.split(), stdout=output) 
            output, error = process.communicate()
    else:
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

    return output, error


def write_pickle(data, pickle_file_name):
    with open(pickle_file_name, 'wb') as file:
        pickle.dump(data, file)
    print_fl(f"Wrote {pickle_file_name}")


def read_pickle(pickle_file_name):
    with open(pickle_file_name, 'rb') as file:
        ret = pickle.load(file)
    return ret

