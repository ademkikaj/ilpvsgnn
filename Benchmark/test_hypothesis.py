import sys
from pyswip import Prolog
import pandas as pd



def test_program(logic_path,dataset_name,relational_path,program, target,background):
        # test the program on the test set
        prolog = Prolog()

        # load the background knowledge
        if background != "None":
            prolog.consult(background)

        # load the program
        prolog.consult(program)

        # convert to prolog file
        # shutil.copy(self.logic_path + f"/tilde/{self.dataset_name}.kb",self.logic_path + f"/tilde/{self.dataset_name}.pl")

        # load the test background knowledge
        prolog.consult(logic_path + f"/{dataset_name}_test.kb")
        
        # load the test queries
        path = relational_path + f"/test/{dataset_name}.csv"
        df = pd.read_csv(relational_path + f"/test/{dataset_name}.csv")
        # sort by problemId
        # if problem it might be here
        #if dataset_name == "train":
            #df = df.sort_values(by=['id'])
        #df = df.sort_values(by=['problemId'])
        
        total = 0
        correct = 0
        index = 0
        pos = 0
        neg = 0
        for index2, row in df.iterrows():
            # create query
            if dataset_name == "sameGen":
                arg1 = row["name1"]
                arg2 = row["name2"]
                query = f"{dataset_name}({index},{arg1},{arg2},Result)"
            elif dataset_name == "colr":
                arg1 = row["id"]
                arg2 = row["node"]
                query = f"{dataset_name}({arg1},{arg2},Result)"
            elif dataset_name == "imdb":
                arg1 = row["person1"]
                arg2 = row["person2"]
                query = f"{dataset_name}({index},{arg1},{arg2},Result)"
            elif dataset_name == "uiversity":
                arg1 = row["student_id"]
                arg2 = row["ranking"]
                query = f"{dataset_name}({arg1},Result,Result2)"
            elif dataset_name == "financial":
                arg1 = row["id"]
                query = f"{dataset_name}({arg1},Result)"
            elif dataset_name == "cancer":
                arg1 = row["id"]
                query = f"{dataset_name}({arg1},Result)"
            # elif dataset_name == "krk":
            #     query = f"{dataset_name}(k{index},Result)"
            elif dataset_name == "cyclic":
                arg1 = row["id"]
                query = f"{dataset_name}({arg1},Result)"
            else:
                query = f"{dataset_name}({index},Result)"
            index += 1
            true_value = row[target]
            query_result = list(prolog.query(query))
            if query_result:
                if query_result[0]['Result'] == true_value:
                    correct += 1
                total += 1
                if query_result[0]['Result'] == 'pos':
                    pos += 1
                else:
                    neg += 1
        
        print(correct/total)
        return
        

if __name__ == '__main__':
    # parse arguments from command line
    logic_path = sys.argv[1]
    dataset_name = sys.argv[2]
    relational_path = sys.argv[3]
    program = sys.argv[4]
    target = sys.argv[5]
    background = sys.argv[6]
    test_program(logic_path,dataset_name,relational_path,program,target,background)
