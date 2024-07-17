import os
from runner import Run 
from bongardGenerator.bongardGenerator import generate_bongard_example
from Benchmark.krk.toLogic import toLogic
from Benchmark.krk.toGraph import toGraph
import pandas as pd


# scalability experiment: increase the number of examples the systems are trained on
dataset_name = "krk"
ilpSettings = {
            "tilde": {
                "node_only": [
                    f"predict({dataset_name}(+B,-C)).\n",
                    "rmode(white_king(+id,-node_id,-x,-y)).\n",
                    "rmode(white_rook(+id,-node_id,-x,-y)).\n",
                    "rmode(black_king(+id,-node_id,-x,-y)).\n",
                    "rmode(edge(+id,-node_id,-node_id)).\n",
                    "rmode(same_x(+X1,#[0,1,2,3,4,5,6,7])).\n",
                    "rmode(same_y(+Y1,#[0,1,2,3,4,5,6,7])).\n",
                    "typed_language(yes).\n",
                    f"type({dataset_name}(id,class)).\n",
                    "type(white_king(id,node_id,x,y)).\n",
                    "type(white_rook(id,node_id,x,y)).\n",
                    "type(black_king(id,node_id,x,y)).\n",
                    "type(edge(id,node_id,node_id)).\n",
                    "type(same_x(x,x)).\n",
                    "type(same_y(y,y)).\n",
                    "auto_lookahead(white_king(Id,Node_id,X,Y),[Node_id]).\n",
                    "auto_lookahead(white_rook(Id,Node_id,X,Y),[Node_id]).\n",
                    "auto_lookahead(black_king(Id,Node_id,X,Y),[Node_id]).\n",
                    "auto_lookahead(edge(Id,Node_id1,Node_id2),[Node_id1,Node_id2]).\n",
                    "auto_lookahead(same_x(X1,X2),[X1,X2]).\n",
                    "auto_lookahead(same_y(Y1,Y2),[Y1,Y2]).\n",
                ],
                "node_edge": [
                    "exhaustive_lookahead(1).\n",
                    f"predict({dataset_name}(+B,-C)).\n",
                    "warmode(white_king(+id,-node_id,#x,#y)).\n",
                    "warmode(white_rook(+id,-node_id,#x,#y)).\n",
                    "warmode(black_king(+id,-node_id,#x,#y)).\n",
                    "warmode(edge(+id,-node_id,-node_id)).\n",
                    "warmode(same_rank(+id,+-node_id,+-node_id)).\n",
                    "warmode(same_file(+id,+-node_id,+-node_id)).\n",

                    "typed_language(yes).\n",
                    f"type({dataset_name}(id,class)).\n",
                    "type(white_king(id,node_id,x,y)).\n",
                    "type(white_rook(id,node_id,x,y)).\n",
                    "type(black_king(id,node_id,x,y)).\n",
                    "type(edge(id,node_id,node_id)).\n",
                    "type(same_rank(id,node_id,node_id)).\n",
                    "type(same_file(id,node_id,node_id)).\n",
                    "type(same_x(x,x)).\n",
                    "type(same_y(y,y)).\n",

                    "rmode(same_x(+X1,#[0,1,2,3,4,5,6,7])).\n",
                    "rmode(same_y(+Y1,#[0,1,2,3,4,5,6,7])).\n",

                    "auto_lookahead(white_king(Id,Node_id,X,Y),[Node_id,X,Y]).\n",
                    "auto_lookahead(white_rook(Id,Node_id,X,Y),[Node_id,X,Y]).\n",
                    "auto_lookahead(black_king(Id,Node_id,X,Y),[Node_id,X,Y]).\n",
                    "auto_lookahead(edge(Id,Node_id1,Node_id2),[Node_id1,Node_id2]).\n",
                    "auto_lookahead(same_rank(Id,Node_id1,Node_id2),[Node_id1,Node_id2]).\n",
                    "auto_lookahead(same_file(Id,Node_id1,Node_id2),[Node_id1,Node_id2]).\n",
                    "auto_lookahead(same_col(X1,X2),[X1,X2]).\n",
                    "auto_lookahead(same_row(Y1,Y2),[Y1,Y2]).\n",
                ],
                "edge_based": [
                    f"predict({dataset_name}(+B,-C)).\n",
                    "warmode(white_king(+id,+node_id,#x,#y)).\n",
                    "warmode(white_rook(+id,+node_id,#x,#y)).\n",
                    "warmode(black_king(+id,+node_id,#x,#y)).\n",
                    "warmode(edge(+id,-node_id,-node_id)).\n",
                    "warmode(same_rank(+id,-node_id,-node_id)).\n",
                    "warmode(same_file(+id,-node_id,-node_id)).\n",
                    "warmode(instance(+id,#piece,-node_id)).\n",
                    "rmode(same_x(+X1,#[0,1,2,3,4,5,6,7])).\n",
                    "rmode(same_y(+Y1,#[0,1,2,3,4,5,6,7])).\n",
                    "typed_language(yes).\n",
                    "type(same_x(x,x)).\n",
                    "type(same_y(y,y)).\n",
                    "auto_lookahead(white_king(Id,Node_id,X,Y),[Node_id,X,Y]).\n",
                    "auto_lookahead(white_rook(Id,Node_id,X,Y),[Node_id,X,Y]).\n",
                    "auto_lookahead(black_king(Id,Node_id,X,Y),[Node_id,X,Y]).\n",
                    "auto_lookahead(edge(Id,Node_id1,Node_id2),[Node_id1,Node_id2]).\n",
                    "auto_lookahead(same_rank(Id,Node_id1,Node_id2),[Node_id1,Node_id2]).\n",
                    "auto_lookahead(same_file(Id,Node_id1,Node_id2),[Node_id1,Node_id2]).\n",
                    "auto_lookahead(instance(Id,white_king,Node_id),[Node_id]).\n",
                    "auto_lookahead(instance(Id,white_rook,Node_id),[Node_id]).\n",
                    "auto_lookahead(instance(Id,black_king,Node_id),[Node_id]).\n",
                    "auto_lookahead(same_x(X1,X2),[X1,X2]).\n",
                    "auto_lookahead(same_y(Y1,Y2),[Y1,Y2]).\n",
                ],
                "Klog": [
                    "predict(krk(+B,-C)).\n",
                    "warmode(white_king(+id,+node_id,#x,#y)).\n",
                    "warmode(white_rook(+id,+node_id,#x,#y)).\n",
                    "warmode(black_king(+id,+node_id,#x,#y)).\n",
                    "warmode(same_rank(+id,-node_id)).\n",
                    "warmode(same_file(+id,-node_id)).\n",
                    "warmode(edge(+id,-node_id,-node_id)).\n",
                    "rmode(same_x(+X1,#[0,1,2,3,4,5,6,7])).\n",
                    "rmode(same_y(+Y1,#[0,1,2,3,4,5,6,7])).\n",
                    "typed_language(yes).\n",
                    "type(same_x(x,x)).\n",
                    "type(same_y(y,y)).\n",
                    "auto_lookahead(white_king(Id,Node_id,X,Y),[Node_id,X,Y]).\n",
                    "auto_lookahead(white_rook(Id,Node_id,X,Y),[Node_id,X,Y]).\n",
                    "auto_lookahead(black_king(Id,Node_id,X,Y),[Node_id,X,Y]).\n",
                    "auto_lookahead(edge(Id,Node_id1),[Node_id1]).\n",
                    "auto_lookahead(same_rank(Id,Node_id1),[Node_id1]).\n",



                ]
            },
            "aleph": {
                "node_only":[
                    ":- modeb(*,white_king(+id, -node_id, #x, #y)).\n",
                    ":- modeb(*,white_rook(+id, -node_id,#x, #y)).\n",
                    ":- modeb(*,black_king(+id, -node_id, #x, #y)).\n",
                    ":- modeb(*,edge(+id, -node_id, -node_id)).\n",   
                    ":- modeb(*,same_x(+x, #[0,1,2,3,4,5,6,7])).\n",
                    ":- modeb(*,same_y(+y, #[0,1,2,3,4,5,6,7])).\n",       
                    f":- determination({dataset_name}/1,white_king/4).\n",
                    f":- determination({dataset_name}/1,white_rook/4).\n",
                    f":- determination({dataset_name}/1,black_king/4).\n",
                    f":- determination({dataset_name}/1,edge/3).\n",
                    f":- determination({dataset_name}/1,same_x/2).\n",
                    f":- determination({dataset_name}/1,same_y/2).\n",
                ],
                "node_edge":[
                    ":- modeb(*,white_king(+id, -node_id, -x, -y)).\n",
                    ":- modeb(*,white_rook(+id, -node_id, -x, -y)).\n",
                    ":- modeb(*,black_king(+id, -node_id, -x, -y)).\n",
                    ":- modeb(*,edge(+id, -node_id, -node_id)).\n",
                    ":- modeb(*,same_rank(+id, -node_id, -node_id)).\n",
                    ":- modeb(*,same_file(+id, -node_id, -node_id)).\n",
                    ":- modeb(*,same_x(+x, #[0,1,2,3,4,5,6,7])).\n",
                    ":- modeb(*,same_y(+y, #[0,1,2,3,4,5,6,7])).\n",
                    f":- determination({dataset_name}/1,white_king/4).\n",
                    f":- determination({dataset_name}/1,white_rook/4).\n",
                    f":- determination({dataset_name}/1,black_king/4).\n",
                    f":- determination({dataset_name}/1,edge/3).\n",
                    f":- determination({dataset_name}/1,same_rank/3).\n",
                    f":- determination({dataset_name}/1,same_file/3).\n",
                    f":- determination({dataset_name}/1,same_x/2).\n",
                    f":- determination({dataset_name}/1,same_y/2).\n",
                ],
                "edge_based":[
                    ":- modeb(*,white_king(+id, -object, #x, #y)).\n",
                    ":- modeb(*,white_rook(+id, -object,#x, #y)).\n",
                    ":- modeb(*,black_king(+id, -object, #x, #y)).\n",
                    ":- modeb(*,edge(+id, -object, -object)).\n",
                    ":- modeb(*,same_rank(+id, -object, -object)).\n",
                    ":- modeb(*,same_file(+id, -object, -object)).\n",
                    ":- modeb(*,instance(+id, white_king, -object)).\n", 
                    ":- modeb(*,instance(+id, white_rook, -object)).\n",
                    ":- modeb(*,instance(+id, black_king, -object)).\n", 
                    ":- modeb(*,same_x(+x, #x)).\n",
                    ":- modeb(*,same_y(+y, #y)).\n",       
                    f":- determination({dataset_name}/1,white_king/4).\n",
                    f":- determination({dataset_name}/1,white_rook/4).\n",
                    f":- determination({dataset_name}/1,black_king/4).\n",
                    f":- determination({dataset_name}/1,edge/3).\n",
                    f":- determination({dataset_name}/1,same_rank/3).\n",
                    f":- determination({dataset_name}/1,same_file/3).\n",
                    f":- determination({dataset_name}/1,instance/3).\n",
                    f":- determination({dataset_name}/1,same_x/2).\n",
                    f":- determination({dataset_name}/1,same_y/2).\n",
                ],
                "Klog":[
                    ":- modeb(*,white_king(+id, -object, #x, #y)).\n",
                    ":- modeb(*,white_rook(+id, -object, #x, #y)).\n",
                    ":- modeb(*,black_king(+id, -object, #x, #y)).\n",
                    ":- modeb(*,edge(+id, -object, -object)).\n",
                    ":- modeb(*,same_rank(+id, -object)).\n",
                    ":- modeb(*,same_file(+id, -object)).\n",     
                    ":- modeb(*,same_x(+x, #x)).\n",
                    ":- modeb(*,same_y(+y, #y)).\n",
                    f":- determination({dataset_name}/1,white_king/4).\n",
                    f":- determination({dataset_name}/1,white_rook/4).\n",
                    f":- determination({dataset_name}/1,black_king/4).\n",
                    f":- determination({dataset_name}/1,edge/3).\n",
                    f":- determination({dataset_name}/1,same_rank/2).\n",
                    f":- determination({dataset_name}/1,same_file/2).\n",
                    f":- determination({dataset_name}/1,same_x/2).\n",
                    f":- determination({dataset_name}/1,same_y/2).\n",
                ]
            },
            "popper":{
                "node_only":[
                    "body_pred(white_king,4).\n",
                    "body_pred(white_rook,4).\n",
                    "body_pred(black_king,4).\n",
                    "body_pred(edge,3).\n",
                    f"type({dataset_name},(id,)).\n",
                    "type(white_king,(id,node_id,x,y)).\n",
                    "type(white_rook,(id,node_id,x,y)).\n",
                    "type(black_king,(id,node_id,x,y)).\n",
                    "type(edge,(id,node_id,node_id)).\n"
                ],
                "node_edge":[
                    "body_pred(white_king,4).\n",
                    "body_pred(white_rook,4).\n",
                    "body_pred(black_king,4).\n",
                    "body_pred(edge,3).\n",
                    "body_pred(same_rank,3).\n",
                    "body_pred(same_file,3).\n",
                    f"type({dataset_name},(id,)).\n",
                    "type(white_king,(id,node_id,x,y)).\n",
                    "type(white_rook,(id,node_id,x,y)).\n",
                    "type(black_king,(id,node_id,x,y)).\n",
                    "type(edge,(id,node_id,node_id)).\n",
                    "type(same_rank,(id,node_id,node_id)).\n",
                    "type(same_file,(id,node_id,node_id)).\n",
                ],
                "edge_based":[
                    "body_pred(white_king,4).\n",
                    "body_pred(white_rook,4).\n",
                    "body_pred(black_king,4).\n",
                    "body_pred(edge,3).\n",
                    "body_pred(same_rank,3).\n",
                    "body_pred(same_file,3).\n",
                    "body_pred(instance,3).\n",
                    f"type({dataset_name},(id,)).\n",
                    "type(white_king,(id,node_id,x,y)).\n",
                    "type(white_rook,(id,node_id,x,y)).\n",
                    "type(black_king,(id,node_id,x,y)).\n",
                    "type(edge,(id,node_id,node_id)).\n",
                    "type(same_rank,(id,node_id,node_id)).\n",
                    "type(same_file,(id,node_id,node_id)).\n",
                    "type(instance,(id,white_king,node_id)).\n",
                    "type(instance,(id,white_rook,node_id)).\n",
                    "type(instance,(id,black_king,node_id)).\n",
                ],
                "Klog":[
                    "body_pred(white_king,4).\n",
                    "body_pred(white_rook,4).\n",
                    "body_pred(black_king,4).\n",
                    "body_pred(edge,3).\n",
                    "body_pred(same_rank,2).\n",
                    "body_pred(same_file,2).\n",
                    f"type({dataset_name},(id,)).\n",
                    "type(white_king,(id,node_id,x,y)).\n",
                    "type(white_rook,(id,node_id,x,y)).\n",
                    "type(black_king,(id,node_id,x,y)).\n",
                    "type(edge,(id,node_id,node_id)).\n",
                    "type(same_rank,(id,node_id)).\n",
                    "type(same_file,(id,node_id)).\n",
                ]
            }
        }

# max bongard examples 392

for num_examples in [100,200,300,400,500,600,700]:

    # generate the relational data
    # output_path = os.path.join("docker","experiment","bongard","relational")
    # generate_bongard_example(num_examples=num_examples,
    #                          object_complexity=10,
    #                          relation_complexity=5,
    #                          rule_complexity=None,
    #                          filename=output_path)

    # only select the num_examples training examples
    df = pd.read_csv(os.path.join("docker","Experiment","krk","relational","original","krk.csv"))
    df = df.sample(num_examples)
    df.to_csv(os.path.join("docker","Experiment","krk","relational","train","krk.csv"),index=False)
    
    bongard_experiment_runner = Run(
        "krk",
        ["node_only","node_edge","edge_based","Klog"],
        "class",
        "id",
        toGraph,
        toLogic,
        ilpSettings,
        "Experiment",
        num_examples
    )
    bongard_experiment_runner.TILDE = True
    bongard_experiment_runner.ALEPH = True
    bongard_experiment_runner.POPPER = False
    bongard_experiment_runner.GNN = True

    
    bongard_experiment_runner.run()
