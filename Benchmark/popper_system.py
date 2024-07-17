import time
import os
import sys
from popper.util import Stats
from popper.util import Settings, format_prog
from popper.loop import learn_solution
import pandas as pd
import subprocess

class Popper:
    def __init__(self, name,relational_path,target):
        self.dataset_name = name
        self.relational_path = relational_path
        self.target = target

    def run(self,representation_path):
        logic_path = os.path.join(representation_path, "logic")
        popper_path = os.path.join(logic_path,"popper")
        start = time.time()
        settings = Settings(kbpath=popper_path,timeout=100,datalog=False,quiet=False,show_stats=True)
        prog, score, stats = learn_solution(settings)
        # print(prog,score,stats)
        # print(stats.show())
        end = time.time()


        if score is not None:
            tp, fn, tn, fp, size = score
            precision = 'n/a'
            if (tp+fp) > 0:
                precision = f'{tp / (tp+fp):0.2f}'
            recall = 'n/a'
            if (tp+fn) > 0:
                recall = f'{tp / (tp+fn):0.2f}'
            train_acc = (tp+tn) / (tp+tn+fp+fn)
        else:
            train_acc = None
        
        runtime = end-start
        if train_acc is not None:
            prog = format_prog(prog)
            self.popper_hyporthesis(prog,os.path.join(popper_path,f"{self.dataset_name}_out.pl"))
            output = self.test_program(representation_path)
            test_acc = output
            representation = os.path.basename(representation_path)
            popper_results = pd.DataFrame([{
                'test_acc': test_acc,
                'runtime': runtime,
                'train_acc': train_acc,
                'system': 'Popper',
                'representation': representation
            }])
        else:
            popper_results = pd.DataFrame([{
                'test_acc': 0,
                'runtime': runtime,
                'train_acc': 0,
                'system': 'Popper',
                'representation': os.path.basename(representation_path)
            }])
        return popper_results

    def popper_hyporthesis(self,prog, output_path):
        new_program = []
        new_program.append(":- style_check(-singleton).")

        for clause in prog:
            if self.dataset_name == "imdb":
                new_clause = prog.replace(f"{self.dataset_name}(A,B)",f"{self.dataset_name}(A,B,pos)")
            elif self.dataset_name == "sameGen":
                new_clause =  prog.replace(f"{self.dataset_name}(A,B,C)",f"{self.dataset_name}(A,B,C,pos)")
            else:
                new_clause =  prog.replace(f"{self.dataset_name}(A)",f"{self.dataset_name}(A,pos)")
            new_clause = new_clause.replace(".", ", !.")
            new_program.append(new_clause)

        # add the negative clause
        if self.dataset_name == "imdb":
            new_program.append(f"{self.dataset_name}(A,B,neg).")
        elif self.dataset_name == "sameGen":
            new_program.append(f"{self.dataset_name}(A,B,C,neg).")
        else:
            new_program.append(f"{self.dataset_name}(A,neg).")

        # write to file
        with open(output_path, "w") as file:
            for line in new_program:
                file.write(line + "\n")
        
        return 

    def test_program(self,input_path):
        logic_path = os.path.join(input_path, "logic")
        program_path = os.path.join(logic_path,"popper",f"{self.dataset_name}_out.pl")
        background = "None"
        output = subprocess.run(["python","Benchmark/test_hypothesis.py",logic_path,self.dataset_name,self.relational_path,program_path,self.target,background],capture_output=True,text=True)
        #print(output)
        accuracy = float(output.stdout)
        return accuracy

