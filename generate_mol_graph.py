#!/usr/bin/env python3

import os
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

from openbabel import openbabel as ob

NAMES     = {1:"H", 6:"C", 7:"N", 8:"O", 16:"S", 17:"Cl"}
#color_map = {'C':'cyan', 'N':'blue', 'O':'red'}


def readfile(f):
    mol = ob.OBMol()
    conv = ob.OBConversion()
    conv.SetInFormat("xyz")

    conv.ReadFile(mol, f)

    return mol

def get_graph(mol):
    G        = nx.Graph()
    atom     = ob.OBAtom()
    numAtoms = mol.NumAtoms()

    for i in range(numAtoms):
        atom = mol.GetAtom(i+1)
        G.add_node(atom.GetIdx(),
                    atomic_num = atom.GetAtomicNum(),
                    is_aromatic = atom.IsAromatic(),
                    atom_symbol = NAMES[atom.GetAtomicNum()])

    #print(G)
    mol_bonds = ob.OBMolBondIter(mol)

    for bond in mol_bonds:
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=bond.GetBondOrder())

    return G

def main():
    X = np.array([])
    shapes = np.array([])

    directory = "xyz"
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        #print(f)

        mol = readfile(f)

        mol_nx = get_graph(mol)

        mol_atom = nx.get_node_attributes(mol_nx, 'atom_symbol')

        mol_colors = []

        #for idx in mol_nx.nodes():
        #    if (mol_nx.nodes[idx]['atom_symbol'] in color_map):
        #        mol_colors.append(color_map[mol_nx.nodes[idx]['atom_symbol']])
        #    else:
        #        mol_colors.append('gray')

    #    nx.draw(caffeine_nx,
    #            labels=caffeine_atom,
    #            with_labels = True,
    #            node_color=caffeine_colors,
    #            node_size=800)

    #    plt.show()

        matrix = nx.to_numpy_matrix(mol_nx)
        x = matrix.flatten()
        x.resize(529)
        X = np.append(X, x)

    X = X.reshape(529, 7211)
    print(X.shape)
    print(X)

    np.save("X.npy", X)

if __name__ == '__main__':
    main()
