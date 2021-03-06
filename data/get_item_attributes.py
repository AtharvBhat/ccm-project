import numpy as np
import os

#change dir to surrent dir
os.chdir(os.getcwd())


def get_item_attributes():
    if os.name == 'nt':
        with open('data/sem_items.txt','r') as fid:
            names_items = np.array([l.strip() for l in fid.readlines()])
        with open('data/sem_relations.txt','r') as fid:
            names_relations = np.array([l.strip() for l in fid.readlines()])
        with open('data/sem_attributes.txt','r') as fid:
            names_attributes = np.array([l.strip() for l in fid.readlines()])
        D = np.loadtxt('data/sem_data.txt')
    else:
        with open('sem_items.txt','r') as fid:
            names_items = np.array([l.strip() for l in fid.readlines()])
        with open('sem_relations.txt','r') as fid:
            names_relations = np.array([l.strip() for l in fid.readlines()])
        with open('sem_attributes.txt','r') as fid:
            names_attributes = np.array([l.strip() for l in fid.readlines()])
        D = np.loadtxt('sem_data.txt')

    nobj = len(names_items)
    nrel = len(names_relations)

    input_pats = D[:,:nobj+nrel]
    attributes = D[:,nobj+nrel:]
    items = input_pats[:, :8]
    relations = input_pats[:, 8:]

    return names_items, names_relations, names_attributes, items, relations, attributes



