import pandas as pd
import numpy as np
from collections import Counter
from itertools import product
from Bio import Entrez, SeqIO, Phylo
import json
import os
from tqdm import tqdm


NUCS = ['A', 'C', 'G', 'T']

############################################ distance measures ############################################

def nuc_distance_metric(seq, k):
    """
    create a vector of the k-nucleotide decomposition (%)
    :param seq: a string under A,T,C,G
    :param k: window
    :return: a vector 1 * 4^k
    """

    combinations = ['{}{}'.format(t1, t2) for t1, t2 in list(product(NUCS, repeat=k))]
    upper_seq = seq.upper()
    n = len(upper_seq)
    counts = []
    for combo in combinations:
        counts.append(upper_seq.count(combo) / (n - k + 1))

    return np.asarray(counts)




def generate_items_latent_matrix(tree, items):
    """

    :param tree:
    :param items:
    :return:
    """

    bacteria_2_nodes = {}


    for node in tree.get_terminals():
        # split via underscore as the trees name are separeted by _
        name = node.name.lower().split('_')[0]
        if name not in bacteria_2_nodes:
            bacteria_2_nodes[name] = [node]
        else:
            bacteria_2_nodes[name].append(node)

    candidates = set(bacteria_2_nodes.keys()).intersection(items['host name'].values)
    print(candidates)

    # search for the sp in each item

    for key in candidates:
        type_sp = [node for node in bacteria_2_nodes[key] if node.name.lower().endswith('type sp')]
        if type_sp != []:
            bacteria_2_nodes[key] = type_sp[0]  # get the type sp as a representative node
        else:
            bacteria_2_nodes[key] = bacteria_2_nodes[key][0]    # get the first node as a representative node

    return bacteria_2_nodes
















############################################ Data preparation ############################################


def parse_host_from_organism(organism):
    """
    parse the organism's description from genebanl
    :param organism: a string containing organism information
    :return: the exact organism and host
    """

    organism = organism.lower()

    if 'phage' in organism:
        host = organism.split('phage')[0].strip()
        organism_type = 'phage'
    elif 'phage' not in organism and 'virus' in organism:
        host = organism.split('virus')[0].strip()
        organism_type = 'virus'
    else:
        host = None
        organism_type = None
        print("Organism {} is not a virus nor a phage".format(organism))

    return host, organism_type



def parse_genbank_recordes(db, out):
    """
    parse the genebank records downloaded directly from NCBI.
    :param db: genebank full records
    :param out:
    :return:
    """

    res = []

    for record in tqdm(SeqIO.parse(db, "gb")):
        annotations = record.annotations

        # ignore cases with multiple accessions numbers (segmented)
        if len(annotations['accessions']) != 1:
            continue

        accession = annotations['accessions'][0]
        host, organism = parse_host_from_organism(annotations['organism'])
        if host == None:
            continue
        taxonomy = annotations['taxonomy']
        order = taxonomy[-2]
        family = taxonomy[-1]

        df = pd.DataFrame({'accession':accession, 'host':host, 'order':order, 'family':family, 'organism':organism}, index=[0])
        res.append(df)

    final = pd.concat(res)
    final.to_csv(os.path.join(out, 'genbank_recoreds_information.csv'), index=False)




def main():
    out = r'/Users/daniellemiller/Google Drive/Msc Bioinformatics/Courses/Recommander Systems/Data Files'
    db = r'/Users/daniellemiller/Google Drive/Msc Bioinformatics/Courses/Recommander Systems/Data Files/phages.gb'

    parse_genbank_recordes(db, out)


if __name__ == "__main__":
    main()










