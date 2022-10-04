#!/usr/bin/env python3

import os
import numpy as np


def get_labels(f):
    lines = open(f, 'r').readlines()

    labels = []

    for line in lines[2:]:
        labels.append(line.split()[0])
    numAtoms = int(lines[0])

    return labels, numAtoms

directory = 'xyz'

d = dict()
nAtoms = np.array([])

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)

    labels, numAtoms = get_labels(f)
    nAtoms = np.append(nAtoms, numAtoms)

    for label in labels:
        d[label] = 1


print(d)
print(np.max(numAtoms))
