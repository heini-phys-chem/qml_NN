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

from sklearn.model_selection import KFold

def get_energies(filename):
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

if __name__ == "__main__":

  data  = get_energies("train.txt")

  mols = []

  for xyz_file in sorted(data.keys()):
    mol = qml.Compound()
    mol.read_xyz("xyz/" + xyz_file + ".xyz")
    mol.properties = data[xyz_file]
    mols.append(mol)

  bags = {
          "H" :  max([mol.atomtypes.count("H" ) for mol in mols]),
          "C" :  max([mol.atomtypes.count("C" ) for mol in mols]),
          "N" :  max([mol.atomtypes.count("N" ) for mol in mols]),
          "O" :  max([mol.atomtypes.count("O" ) for mol in mols]),
          "S" :  max([mol.atomtypes.count("S" ) for mol in mols]),
          "Cl" :  max([mol.atomtypes.count("Cl" ) for mol in mols]),
  }

  for mol in mols:
    mol.generate_bob(asize=bags)

  ll = [1e-7, 1e-11, 1e-15]
  sigma = [0.1*2**i for i in range(1,20)]

  X      = np.asarray([mol.representation for mol in mols])
  Yprime = np.asarray([ mol.properties for mol in mols ])

  kf = KFold(n_splits=5)
  kf.get_n_splits(X)

  print(kf)
  for j in range(len(sigma)):
    for l in ll:
      maes = []
      for train_index, test_index in kf.split(X):
        K      = gaussian_kernel(X[train_index], X[train_index], sigma[j])
        K_test = gaussian_kernel(X[train_index], X[test_index],  sigma[j])

        Y = Yprime[train_index]

        C = deepcopy(K)
        C[np.diag_indices_from(C)] += l

        alpha = cho_solve(C, Y)

        Yss  = np.dot(K_test.T, alpha)
        diff = Yss- Yprime[test_index]
        mae  = np.mean(np.abs(diff))
        maes.append(mae)

      print( str(l) + ',' + str(sigma[j]) + "," + str(sum(maes)/len(maes)) )
