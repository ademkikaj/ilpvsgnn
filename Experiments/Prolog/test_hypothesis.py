import sys
from pyswip import Prolog
import pandas as pd



def test_program(logic_path,dataset_name,relational_path,program):
        # test the program on the test set
        prolog = Prolog()
 
        # load the program
        prolog.consult(program)

        # convert to prolog file
        # shutil.copy(self.logic_path + f"/tilde/{self.dataset_name}.kb",self.logic_path + f"/tilde/{self.dataset_name}.pl")


        # load the test background knowledge
        
        prolog.consult(logic_path + f"/{dataset_name}.pl")
        
        # load the test queries
        df = pd.read_csv(relational_path + f"/test/{dataset_name}.csv")

        
        total = 0
        correct = 0
        for _, row in df.iterrows():
            # create the query
            query = f"{dataset_name}({row['problemId']},Result)"
            true_value = row['class']
            query_result = list(prolog.query(query))
            if query_result:
                if query_result[0]['Result'] == true_value:
                    correct += 1
                total += 1
        
        print(correct/total)
        return
        

if __name__ == '__main__':
    # parse arguments from command line
    logic_path = sys.argv[1]
    dataset_name = sys.argv[2]
    relational_path = sys.argv[3]
    program = sys.argv[4]
    test_program(logic_path,dataset_name,relational_path,program)
