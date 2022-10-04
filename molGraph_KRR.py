#!/usr/bin/env python3

import sys
import time
from datetime import datetime
import random
#import cPickle
import numpy as np
from copy import deepcopy
import qml
from qml.representations import *
from qml.kernels import gaussian_kernel
from qml.kernels import laplacian_kernel
from qml.math import cho_solve
import itertools
from time import time

from openbabel import openbabel as ob
import networkx as nx

NAMES     = {1:"H", 6:"C", 7:"N", 8:"O", 16:"S", 17:"Cl"}


def get_energies(filename):
  """ returns dic with properties for xyz files
  """
  f = open(filename, "r")
  lines = f.readlines()
  f.close()

  properties = dict()

  for line in lines:
    tokens = line.split()
    xyz_name = tokens[0]
    property = float(tokens[1])
    properties[xyz_name] = property

  return properties


def generate_graph(f):
    mol = ob.OBMol()
    conv = ob.OBConversion()
    conv.SetInFormat("xyz")

    conv.ReadFile(mol, f)

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

    mol_nx = G
    matrix = nx.to_numpy_matrix(mol_nx)

    x = matrix.flatten()
    x.resize(529)

    return x


if __name__ == "__main__":

  data_train  = get_energies("train.txt")
  data_test   = get_energies( "test.txt")


  mols = []
  mols_test = []

  for xyz_file in sorted(data_train.keys()):
    mol = qml.Compound()
    mol.read_xyz("xyz/" + xyz_file + ".xyz")
    mol.properties = data_train[xyz_file]
    name = xyz_file
    mols.append(mol)

  for xyz_file in sorted(data_test.keys()):
    mol = qml.Compound()
    mol.read_xyz("xyz/" + xyz_file + ".xyz")
    mol.properties = data_test[xyz_file]
    name = xyz_file
    mols_test.append(mol)


  N = [435, 875, 1750, 3500, 6989]
  total = len(mols)
  nModels = 10
  ll = [1e-3]
  sigma = [12.8]

  X        = np.asarray([generate_graph(mol.name) for mol in mols])
  X_test   = np.asarray([generate_graph(mol.name) for mol in mols_test])

  Yprime = np.asarray([ mol.properties for mol in mols ])
  Y_test = np.asarray([ mol.properties for mol in mols_test ])

  random.seed(667)

  for j in range(len(sigma)):
    K      = laplacian_kernel(X, X, sigma[j])
    K_test = laplacian_kernel(X, X_test, sigma[j])

    for l in ll:
      print()
      for train in N:
        maes = []
        for i in range(nModels):
          split = np.array(list(range(total)))
          random.shuffle(split)

          training_index  = split[:train]

          Y = Yprime[training_index]

          C = K[training_index[:,np.newaxis],training_index]
          C[np.diag_indices_from(C)] += l
          alpha = cho_solve(C, Y)

          Yss = np.dot((K_test[training_index]).T, alpha)
          diff = Yss  - Y_test
          mae = np.mean(np.abs(diff))
          maes.append(mae)
          s = np.std(maes)/np.sqrt(nModels)

        print(str(l) + "\t" + str(sigma[j]) +  "\t" + str(train) + "\t" + str(sum(maes)/len(maes)) + " " + str(s))
