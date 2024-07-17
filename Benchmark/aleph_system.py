import os
import subprocess
import pandas as pd
import re

class Aleph:

    def __init__(self, name, relational_path,target):
        self.dataset_name = name
        self.relational_path = relational_path
        self.target = target
    
    def run(self,representation_path):
        
        aleph_path = os.path.join(representation_path, "logic", "aleph")
        # aleph_model = aleph_learn(
        #     file = os.path.join(aleph_path, "background.pl"),
        #     settings= os.path.join(aleph_path, "modes.pl"),
        #     positive_example= os.path.join(aleph_path, "pos_example.f"),
        #     negative_example= os.path.join(aleph_path, "neg_example.n"),
        #     test_size=0.0,
        #     shuffle=False
        # )
        output = subprocess.run(["python","Benchmark/run_aleph.py",aleph_path],capture_output=True,text=True)
        # output is a string consisting of (hypothesis, runtime, train_acc), extract it 
        print(output)
        hypothesis = output.stdout.split()[0][1:-1]
        runtime = float(output.stdout.split()[1][:-1])
    
        output_path = os.path.join(aleph_path, f"{self.dataset_name}_out.pl")
        self.aleph_hypothesis(hypothesis,output_path)

        runtime = runtime
        train_acc = 0.0

        # test the program
        test_acc = self.test_program(representation_path)

        print("test accuracy: ", test_acc)
        print("hypothesis: ", hypothesis)

        results = pd.DataFrame([{
            'test_acc': test_acc,
            'runtime': runtime,
            'train_acc': train_acc,
            'system': 'Aleph',
            'representation': os.path.basename(representation_path)
        }])
        return results
    

    def check_if_number(self, string):
        start_idx = string.find("(") + 1
        end_idx = string.find(")") 
        if start_idx != 0 and end_idx != -1:
            number = string[start_idx:end_idx]
            if number.isdigit():
                return True
        return False

    
    def aleph_hypothesis(self, hypothesis,output_path):
        program = hypothesis
        program = program.strip("'").split("\\n")
        # add an extra argument to the program clauses
        new_program = []

        pattern = r'(' + re.escape(self.dataset_name) + r'\()([A-Za-z_][A-Za-z0-9_]*)\)'

        # add style check singleton
        new_program.append(":- style_check(-singleton).")
        for clause in program:
            clause = clause.replace(".", ", !.")
            if self.dataset_name == "imdb":
                new_program.append(clause.replace(f"{self.dataset_name}(A,B)",f"{self.dataset_name}(A,B,pos)"))
            elif self.dataset_name == "sameGen":
                new_program.append(clause.replace(f"{self.dataset_name}(A,B,C)",f"{self.dataset_name}(A,B,C,pos)"))
            elif self.dataset_name == "colr":
                new_program.append(clause.replace(f"{self.dataset_name}(A,B)",f"{self.dataset_name}(A,B,pos)"))
            else:
                if clause != "None" and not self.check_if_number(clause):
                    # if it is a number between the brackets then skip the clause
                    modified_clause = re.sub(pattern, r'\1\2,pos)', clause)
                    new_program.append(modified_clause)
                #new_program.append(clause.replace(f"{self.dataset_name}(A)",f"{self.dataset_name}(A,pos)"))
        # add the negative clause
        if self.dataset_name == "imdb":
            new_program.append(f"{self.dataset_name}(A,B,neg).")
        elif self.dataset_name == "sameGen":
            new_program.append(f"{self.dataset_name}(A,B,C,neg).")
        elif self.dataset_name == "colr":
            new_program.append(f"{self.dataset_name}(A,B,neg).")
        else:
            new_program.append(f"{self.dataset_name}(A,neg).")

        self.hypothesis = new_program
        
        with open(output_path, "w") as file:
            for line in new_program:
                file.write(line + "\n")
        return
    
    def test_program(self,tilde_input_path):
        logic_path = os.path.join(tilde_input_path, "logic")
        program_path = os.path.join(logic_path,"aleph",f"{self.dataset_name}_out.pl")
        
        if os.path.exists(os.path.join(logic_path,"aleph",f"extra_bg.pl")):
            background = os.path.join(logic_path,"aleph",f"extra_bg.pl")
        else:
            background = "None"
        output = subprocess.run(["python","Benchmark/test_hypothesis.py",logic_path,self.dataset_name,self.relational_path,program_path,self.target,background],capture_output=True,text=True)
        accuracy = float(output.stdout)
        return accuracy