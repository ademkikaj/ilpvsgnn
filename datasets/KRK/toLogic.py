import pandas as pd
import os
from pathlib import Path
import glob


white_kings = pd.read_csv('datasets/KRK/Relational/white_king.csv')
white_rooks = pd.read_csv('datasets/KRK/Relational/white_rook.csv')
black_kings = pd.read_csv('datasets/KRK/Relational/black_king.csv')
classes = pd.read_csv('datasets/KRK/Relational/class.csv')

amount_of_examples = len(classes)

# write the models
file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/docker/KRK/NoModels/krk.kb"

with open(file_path,'w') as file:

    for i in range(amount_of_examples):
        id = str(i)
        file.write('\n')

        result = "neg" if classes.iloc[i]['class'] == "illegal" else "pos"
        file.write("krk("+ id + "," + result + ").\n")

        white_king = "white_king(" + id + "," + str(white_kings.iloc[i]['file']) + "," + str(white_kings.iloc[i]['rank']) + ").\n"
        file.write(white_king)

        white_rook = "white_rook(" + id + "," + str(white_rooks.iloc[i]['file']) + "," + str(white_rooks.iloc[i]['rank']) + ").\n"
        file.write(white_rook)

        black_king = "black_king(" + id + "," + str(black_kings.iloc[i]['file']) + "," + str(black_kings.iloc[i]['rank']) + ").\n"
        file.write(black_king)



file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/KRK/Popper/exs.pl"

with open(file_path,'w') as file:
    for i in range(amount_of_examples):
        id = str(i)

        WK_file = str(white_kings.iloc[i]['file'])
        WK_rank = str(white_kings.iloc[i]['rank'])

        WR_file = str(white_rooks.iloc[i]['file'])
        WR_rank = str(white_rooks.iloc[i]['rank'])

        BK_file = str(black_kings.iloc[i]['file'])
        BK_rank = str(black_kings.iloc[i]['rank'])

        result = "neg" if classes.iloc[i]['class'] == "illegal" else "pos"
        file.write(result + "(" + "krk(" + str(WK_file) +","+str(WK_rank)+","+str(WR_file)+","+str(WR_rank) + "," + str(BK_file) + "," + str(BK_rank) + ")).\n")


file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/KRK/Popper/bk.pl"

# with open(file_path,'w') as file:
    
#     for i in range(len(white_kings)):
#         id = white_kings.iloc[i]['id']
#         id = str(id)
#         file.write("white_king(" +id + ','+ str(white_kings.iloc[i]['file']) + "," + str(white_kings.iloc[i]['rank']) + ").\n")
    
#     for i in range(len(white_rooks)):
#         id = white_rooks.iloc[i]['id']
#         id = str(id)
#         file.write("white_rook(" +id + ','+ str(white_rooks.iloc[i]['file']) + "," + str(white_rooks.iloc[i]['rank']) + ").\n")
    
#     for i in range(len(black_kings)):
#         id = black_kings.iloc[i]['id']
#         id = str(id)
#         file.write("black_king(" +id + ','+ str(black_kings.iloc[i]['file']) + "," + str(black_kings.iloc[i]['rank']) + ").\n")




####----------- Create popper files -----------####

amount_of_examples = 10
pos_n = amount_of_examples//2
neg_n = amount_of_examples - pos_n

### exs.pl ###
""" Has the form pos(f(b1)) or neg(f(b1)) """
# sample random examples
pos_examples = classes[classes['class'] == 'legal']
pos_train  = pos_examples.sample(n=pos_n)
pos_test = pos_examples.drop(pos_train.index)
neg_examples = classes[classes['class'] == 'illegal'].sample(n=neg_n)
neg_train = neg_examples.sample(n=neg_n)
neg_test = neg_examples.drop(neg_train.index)

ids = pos_train['id'].tolist() + neg_train['id'].tolist()

folder_path = Path(f"/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/KRK/Popper/Popper{amount_of_examples}")
folder_path.mkdir(exist_ok=True)

file_path = f"/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/KRK/Popper/Popper{amount_of_examples}/exs.pl"
with open(file_path,'w') as file:
    for i in range(len(pos_train)):
        id = f"p{pos_train.iloc[i]['id']}"
        file.write("pos(" + "f(" + id + ")).\n")

    for i in range(len(neg_train)):
        id = f"p{neg_train.iloc[i]['id']}"
        file.write("neg(" + "f(" + id + ")).\n")

### bk.pl ###
""" generally has the form square(id,(file,rank),color,piece)"""
file_path = f"/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/KRK/Popper/Popper{amount_of_examples}/bk.pl"
with open(file_path,'w') as file:

    file.write("king(k).\n")
    file.write("rook(r).\n")
    file.write("white(w).\n")
    file.write("black(b).\n")
    file.write("distance((X1,Y1),(X2,Y2),D) :- D1 is abs(X1-X2), D2 is abs(Y1-Y2), D is max(D1,D2).\n")
    file.write("one(1).\n")

    file.write("\n")

    for i in ids:
        id = f"p{i}"
        file.write("square(" + id + "," + f"({white_kings[white_kings['id'] == i]['file'].values[0]},{white_kings[white_kings['id'] == i]['rank'].values[0]})" + ",w,k).\n")
        file.write("square(" + id + "," + f"({white_rooks[white_rooks['id'] == i]['file'].values[0]},{white_rooks[white_rooks['id'] == i]['rank'].values[0]})" + ",w,r).\n")
        file.write("square(" + id + "," + f"({black_kings[black_kings['id'] == i]['file'].values[0]},{black_kings[black_kings['id'] == i]['rank'].values[0]})" + ",b,k).\n")
    
    
### bias.pl ###
file_path = f"/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/KRK/Popper/Popper{amount_of_examples}/bias.pl"



####----------- Create TILDE files -----------####

### krk.kb ###
file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/docker/KRK/NoModels2/krk.kb"

with open(file_path,'w') as file:
    for i in range(len(classes)):
        id = str(i)

        WK_file = str(white_kings.iloc[i]['file'])
        WK_rank = str(white_kings.iloc[i]['rank'])

        WR_file = str(white_rooks.iloc[i]['file'])
        WR_rank = str(white_rooks.iloc[i]['rank'])

        BK_file = str(black_kings.iloc[i]['file'])
        BK_rank = str(black_kings.iloc[i]['rank'])

        result = "neg" if classes.iloc[i]['class'] == "illegal" else "pos"
        file.write("krk("+ id + "," + result + ").\n")

        file.write(f"square({id},{WK_file},{WK_rank},w,k).\n")
        file.write(f"square({id},{WR_file},{WR_rank},w,r).\n")
        file.write(f"square({id},{BK_file},{BK_rank},b,k).\n")



