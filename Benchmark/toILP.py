import pandas as pd
import os
import shutil
import re

class toILP:
    # Converter class used to convert relational data to ILP format

    def __init__(self, relational_path,logic_path,dataset_name):
        self.relational_path = relational_path
        self.relational_test_path = relational_path + "/test/"
        self.logic_path = logic_path
        self.dataset_name = dataset_name
        self.aleph_modes_path = "Experiments/Aleph/aleph_modes_bongard.pl"
        self.tilde_settings_path = f"Experiments/Tilde/{self.dataset_name}.s"

        # make the logic directories
        ilpSystems = ["tilde","aleph","popper"]
        for system in ilpSystems:
            if not os.path.exists(self.logic_path + "/" + system):
                os.makedirs(self.logic_path + "/" + system)
    
    
    def logic_parsing(self,logic_file_path):
        kb = {}        
        pattern = r"(\w+)\(([^)]+)\)"
        with open(logic_file_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                match = re.match(pattern,line)
                if match:
                    predicate = match.group(1)
                    args = match.group(2).split(',')
                    if predicate in kb:
                        kb[predicate].append(args)
                    else:
                        kb[predicate] = [args]
                else:
                    print("No regex match found in line: ", line)
        return kb
    
    def getDataframes(self,test=False):
        # import all csv files in the relational path to pandas dataframe    
        path = self.relational_path if not test else self.relational_test_path
        dataframes = {}
        for file in os.listdir(path):
            if file.endswith(".csv"):
                dataframes[file.split('.')[0]] = pd.read_csv(os.path.join(path, file))
        return dataframes
    
    def classifyExamples(self,dataframe):
        # classify the examples into positive and negative
        pos_examples, neg_examples = [],[]
        for i in range(len(dataframe)):
            row_list = dataframe.iloc[i].tolist()
            row_list = [str(x) for x in row_list]
            # check the last argument for the class
            if row_list[-1].strip() == "pos":
                pos_examples.append(self.dataset_name + "(" + ",".join(row_list[:-1]) + ").")
            else:
                neg_examples.append(self.dataset_name + "(" + ",".join(row_list[:-1]) + ").")
        return pos_examples, neg_examples
    
    def to_kb(self,output_path,test=False):
        dataframes = self.getDataframes(test=test)
        background = []
        for key in dataframes:
            df = dataframes[key]
            for i in range(len(df)):
                row_list = df.iloc[i].tolist()
                # turn all elements in row_list to strings if not already
                row_list = [str(x) for x in row_list]
                background.append(key + "(" + ",".join(row_list) + ").")
        
        with open(output_path, "w") as file:
            for line in background:
                if test:
                    if self.dataset_name not in line:
                        file.write(line + "\n")
                else:
                    file.write(line + "\n")

        return

    def toTilde(self,given_settings):
        # convert relational data to Tilde format
        # Consists of background, examples and settings file
        background, examples = [],[]
        dataframes = self.getDataframes()


        for key in dataframes:
            df = dataframes[key]
            for i in range(len(df)):
                row_list = df.iloc[i].tolist()
                # turn all elements in row_list to strings if not already
                row_list = [str(x) for x in row_list]
                background.append(key + "(" + ",".join(row_list) + ").")
        
        with open(self.logic_path + f"/tilde/{self.dataset_name}.kb", "w") as file:
            for line in background:
                file.write(line + "\n")

        settings = []
        settings.append("output_options([c45,c45c,c45e,lp,prolog]).\n")
        settings.append("use_packs(ilp).\n")

        settings += given_settings

        settings.append("write_predictions([testing, distribution]).\n")
        settings.append("combination_rule([product, sum]).\n")
        settings.append("execute(t).\n")
        settings.append("execute(q).\n")

        with open(self.logic_path + f"/tilde/{self.dataset_name}.s", "w") as file:
            for line in settings:
                file.write(line )
        
        return 

    def logicToTilde(self,logic_file_path, givesettings):

        # copy the logic file to the tilde path
        shutil.copy(logic_file_path,self.logic_path + f"/tilde/{self.dataset_name}.kb")

        # add a settings file
        settings = []
        settings.append("output_options([c45,c45c,c45e,lp,prolog]).\n")
        settings.append("use_packs(ilp).\n")
        # settings.append("max_lookahead(2).\n")
        # settings.append("exhaustive_lookahead(1).\n")
        # settings.append("query_batch_size(50000).\n")


        settings += givesettings

        settings.append("write_predictions([testing, distribution]).\n")
        settings.append("combination_rule([product, sum]).\n")
        settings.append("execute(t).\n")
        settings.append("execute(q).\n")

        with open(self.logic_path + f"/tilde/{self.dataset_name}.s", "w") as file:
            for line in settings:
                file.write(line)

        return

    
    def toAleph(self,modes_aleph):
        # Convert relational data to Aleph format
        # Consists of background knowledge, modes, pos_examples and neg_examples
        background = []
        modes = []
        pos_examples = []
        neg_examples = []
        # import all csv files in the relational path to pandas dataframe
        dataframes = self.getDataframes()

        # go over all predicates and convert to Aleph format
        for key in dataframes:
            # these are the examples
            if key == self.dataset_name:
                pos_examples, neg_examples = self.classifyExamples(dataframes[key])
            # these are the background knowledge
            else:
                df = dataframes[key]
                for i in range(len(df)):
                    row_list = df.iloc[i].tolist()
                    # turn all elements in row_list to strings if not already
                    row_list = [str(x) for x in row_list]
                    background.append(key + "(" + ",".join(row_list) + ").")
                    
            
        # copy the modes file to the logic path
        #shutil.copy(self.aleph_modes_path,self.logic_path + "/aleph/modes.pl")
                    
        modes.append(":- aleph_set(i,2).\n")
        modes.append(":- aleph_set(verbosity,1).\n")
        modes.append(":- aleph_set(clauselength,5).\n")
        #modes.append(":- aleph_set(minacc,0.2)")
        modes.append(":- aleph_set(minpos,2).\n")
        modes.append(":- aleph_set(nodes,50000).\n")
        modes.append(":- aleph_set(noise,5).\n")
        modes.append(":- aleph_set(c,3).\n")

        modes += modes_aleph

        # write to file 
        # write background knowledge to file
        with open(self.logic_path + "/aleph/background.pl", "w") as file:
            for line in background:
                file.write(line + "\n")
        # write positive examples
        with open(self.logic_path + "/aleph/pos_example.f", "w") as file:
            for line in pos_examples:
                file.write(line + "\n")
        # write negative examples
        with open(self.logic_path + "/aleph/neg_example.n", "w") as file:
            for line in neg_examples:
                file.write(line + "\n")
        with open(self.logic_path + "/aleph/modes.pl", "w") as file:
            for line in modes:
                file.write(line)
        return

    def logicToAleph(self,logic_file_path,label, given_settings):
        background,pos_examples,neg_examples = [],[],[]

        # kb = self.logic_parsing(logic_file_path)
        # # add discontinuous rules to the background
        # for key in kb:
        #     background.append(f":- discontiguous {key}/{len(kb[key][0])}.")

        # if "bongard" == self.dataset_name and "edge_based" in logic_file_path:
        #     background.append("triangle(triangle).\n")
        #     background.append("circle(circle).\n")
        #     background.append("square(square).\n")

        # num_examples = 20
        # current_examples = 0

        with open(logic_file_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith(label):
                    if "pos" in line:
                        new_line = line.split(",pos")[0]
                        pos_examples.append(new_line + ').')
                        # current_examples += 1
                        # if current_examples > num_examples:
                        #     # remove the last example
                        #     pos_examples.pop()
                        #     break
                    elif "neg" in line:
                        new_line = line.split(",neg")[0]
                        neg_examples.append(new_line + ').')

                        # current_examples += 1
                        # if current_examples > num_examples:
                        #     # remove the last example
                        #     neg_examples.pop()
                        #     break
                    
                else:
                    background.append(line.strip())
        
        # bias file
        settings = []
        settings.append(":- use_module(library(aleph)).\n")
        settings.append(":- if(current_predicate(use_rendering/1)).\n")
        settings.append(":- use_rendering(prolog).\n")
        settings.append(":- endif.\n")
        settings.append(":- aleph.\n")
        settings.append(":- style_check(-discontiguous).\n")
        #settings.append(":- aleph_set(clauselength,7).\n")
        #settings.append(":- aleph_set(c,3).\n")
        #settings.append(":- aleph_set(i,6).\n")
        settings.append(":- aleph_set(verbosity,1).\n")
        #settings.append(":- aleph_set(lookahead,5).\n")
        #settings.append(":- aleph_set(minacc,0.05).\n")
        settings.append(":- aleph_set(minpos,2).\n")
        settings.append(":- aleph_set(nodes,100000).\n")
        #settings.append(":- aleph_set(noise,5).\n")
        if self.dataset_name == "sameGen":
            settings.append(f":- modeh(1,{self.dataset_name}(+id,+name,+name)).\n")
        elif self.dataset_name == "color":
            settings.append(f":- modeh(1,{self.dataset_name}(+id,+node_id)).\n")
        else:
            settings.append(f":- modeh(1,{self.dataset_name}(+id)).\n")
            

        settings += given_settings

        # write to files
        with open(self.logic_path + "/aleph/background.pl", "w") as file:
            for line in background:
                file.write(line + "\n")
        with open(self.logic_path + "/aleph/pos_example.f", "w") as file:
            for line in pos_examples:
                file.write(line + "\n")
        with open(self.logic_path + "/aleph/neg_example.n", "w") as file:
            for line in neg_examples:
                file.write(line + "\n")
        with open(self.logic_path + "/aleph/modes.pl", "w") as file:
            for line in settings:
                file.write(line)
        return
    

    def toMetagol(self):
        # Relational data to Metagol format
        # Consists of background, meta, neg_examples, pos_examples
        background, meta = [],[]

        dataframes = self.getDataframes()

        keys = []
        for key in dataframes:
            # these are the examples
            if key == self.dataset_name:
                pos_examples, neg_examples = self.classifyExamples(dataframes[key])
            # these are the background knowledge
            else:
                df = dataframes[key]
                for i in range(len(df)):
                    row_list = df.iloc[i].tolist()
                    # turn all elements in row_list to strings if not already
                    row_list = [str(x) for x in row_list]
                    background.append(key + "(" + ",".join(row_list) + ").")
                keys.append(key)
        
        # meta file
        meta.append('%%%%%%%%%% tell metagol to use the BK %%%%%%%%%%')
        for key in keys:
            meta.append(f"body_pred({key}/{len(dataframes[key].columns)}).")
        meta.append("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ")
        meta.append("%%%%%%%%%%%%%%%%%%%% metarules %%%%%%%%%%%%%%%%%%%%")
        metarules = [
            "metarule([P,Q], [P,A],[[Q,A]]).",
            "metarule([P,Q,R], [P,A], [[Q,A],[R,A]]).",
            "metarule([P,Q,R], [P,A], [[Q,A,B],[R,B]]).",
            "metarule([P,Q], [P,A,B], [[Q,A,B]]).",
            "metarule([P,Q,R], [P,A,B], [[Q,A,B],[R,B]]).",
            "metarule([P,Q,X], [P,A], [[Q,A,X]]).",
            "metarule([P,Q,X], [P,A,B], [[Q,A,B,X]])."
        ]
        metarules.extend([
            "metarule([P,Q], [P,A,B], [[Q,A,B]]).",
            "metarule([P,Q], [P,A,B], [[Q,B,A]]).",
            "metarule([P,Q,R], [P,A,B], [[Q,A],[R,A,B]]).",
            "metarule([P,Q,R], [P,A,B], [[Q,A,B],[R,B]]).",
            "metarule([P,Q,R], [P,A,B], [[Q,A,C],[R,C,B]])."
        ])
        meta.extend(metarules)

        # background file 
        background = ["%%%%%%%%%%%%%% background knowledge %%%%%%%%%%%%%%"] + background + ["%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n"]


        # write the files
        with open(self.logic_path + "/metagol/BK.pl", "w") as file:
            for line in background:
                file.write(line + "\n")
        with open(self.logic_path + "/metagol/meta.pl", "w") as file:
            for line in meta:
                file.write(line + "\n")
        with open(self.logic_path + "/metagol/pos_example.f", "w") as file:
            for line in pos_examples:
                file.write(line + "\n")
        with open(self.logic_path + "/metagol/neg_example.n", "w") as file:
            for line in neg_examples:
                file.write(line + "\n")
        
        return

    def logicToPopper(self,logic_file_path,label,bias_given):
        background, examples, bias = [],[],[]

        # kb = self.logic_parsing(logic_file_path)
        # # add discontinuous rules to the background
        # for key in kb:
        #     background.append(f":- discontiguous {key}/{len(kb[key][0])}.")

        amount_examples = 20
        current_examples = 0

        with open(logic_file_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith(label):
                    if "pos" in line:
                        # the string without ",pos"
                        nr = line.split(",pos")[0]
                        examples.append("pos(" + nr + ")).")
                    else:
                        nr = line.split(",neg")[0]
                        examples.append("neg(" + nr + ")).")
                    current_examples += 1
                    if current_examples > amount_examples:
                        # remove the last example
                        examples.pop()
                        break
                else:
                    background.append(line.strip())
        
        # bias file
        bias.append('max_vars(8).\n')
        bias.append('max_body(6).\n')
        #bias.append('max_clauses(1).\n')
        if self.dataset_name == "sameGen":
            bias.append('head_pred(sameGen,3).\n')
        else:
            bias.append('head_pred(' + self.dataset_name + ',1).\n')

        bias += bias_given

        # write to files
        with open(self.logic_path + "/popper/bk.pl", "w") as file:
            for line in background:
                file.write(line + "\n")
        with open(self.logic_path + "/popper/exs.pl", "w") as file:
            file.write(":- discontiguous pos/1. \n")
            file.write(":- discontiguous neg/1. \n")
            for line in examples:
                file.write(line + "\n")
        with open(self.logic_path + "/popper/bias.pl", "w") as file:
            for line in bias:
                file.write(line)
        return
        
    
    def toPopper(self,bias):
        # relational data to Popper format
        # Consists of background(bk.pl), examples(examples.pl) and bias file(bias.pl)
        background, examples = [],[]
        dataframes = self.getDataframes()
        bias = bias
        keys = []
        for key in dataframes:
            # these are the examples
            if key == self.dataset_name:
                for i in range(len(dataframes[key])):
                    row_list = dataframes[key].iloc[i].tolist()
                    row_list = [str(x) for x in row_list]
                    if row_list[-1].strip() == "pos":
                        examples.append("pos(" + key + "(" + ",".join(row_list[:-1]) + ")).")
                    else:
                        examples.append("neg(" + key + "(" + ",".join(row_list[:-1]) + ")).")
            # these are the background knowledge
            else:
                df = dataframes[key]
                for i in range(len(df)):
                    row_list = df.iloc[i].tolist()
                    # turn all elements in row_list to strings if not already
                    row_list = [str(x) for x in row_list]
                    background.append(key + "(" + ",".join(row_list) + ").")
                keys.append(key)
        
        # bias file
       

        # write the files
        with open(self.logic_path + "/popper/bk.pl", "w") as file:
            for line in background:
                file.write(line + "\n")
        with open(self.logic_path + "/popper/exs.pl", "w") as file:
            file.write(":- discontiguous pos/1. \n")
            file.write(":- discontiguous neg/1. \n")
            for line in examples:
                file.write(line + "\n")
        with open(self.logic_path + "/popper/bias.pl", "w") as file:
            for line in bias:
                file.write(line)
        return

    def toProlog(self):
        # convert relational data to prolog format
        # Consists of single prolog file
        prolog = []

        dataframes = self.getDataframes(test=True)

        for key in dataframes:
            if key != self.dataset_name:
                df = dataframes[key]
                for i in range(len(df)):
                    row_list = df.iloc[i].tolist()
                    # turn all elements in row_list to strings if not already
                    row_list = [str(x) for x in row_list]
                    prolog.append(key + "(" + ",".join(row_list) + ").")
        
        # write to file
        with open(self.logic_path + f"/{self.dataset_name}.pl", "w") as file:
            for line in prolog:
                file.write(line + "\n")
        
        return
        


    
