import sys
import os
import pandas as pd
from utils.bongardGenerator.bongardGenerator import generate_bongard_example
from Benchmark.aleph_system import Aleph
from Benchmark.popper_system import Popper
from Benchmark.tilde import Tilde


ALEPH = True
POPPER = False
TILDE = False

dataset_name = "cyclic"
repr = "node_edge"
target = "class"
relational_path = os.path.join("docker","Benchmark",dataset_name,"relational")
representation_path = os.path.join("docker","Benchmark",dataset_name,repr)

if ALEPH:
    Al = Aleph(dataset_name,relational_path,target)
    results = Al.run(representation_path)
    print(results)

#docker/Benchmark/krk/relational/test/krk.csv
if POPPER:  
    pop = Popper(dataset_name,relational_path,target)
    accuracy = pop.test_program(representation_path)
    print(accuracy)
    #results = pop.run(representation_path)
    #print(results)


if TILDE:
    Tilde = Tilde(dataset_name,relational_path,target)
    results = Tilde.run(representation_path)
    #accuracy = Tilde.test_program(representation_path)
    print(results)