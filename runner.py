import os
from Benchmark.benchmark import Benchmark
import importlib
import torch
import pandas as pd
import yaml

from Benchmark.tilde import Tilde
from Benchmark.aleph_system import Aleph
from Benchmark.popper_system import Popper

from Benchmark.toILP import toILP


class Run:
    
    def __init__(self,dataset_name,
                 representations,
                 target="class",
                 problem_id="id",
                 base_folder = "datasets",
                 result_id = None,
                 split_data = True):
        
        self.dataset_name = dataset_name
        self.representations = representations
        self.target = target
        self.problem_id = problem_id
        self.ilp_systems = ["tilde","aleph","popper"]
        self.benchmark = Benchmark()

        self.base_folder =  base_folder
        self.result_id = result_id

        # create the folders
        self.create_folders()

        # create the converters
        toGraphMod = importlib.import_module(f"Benchmark.{self.dataset_name}.toGraph")
        GraphConverter = getattr(toGraphMod, "toGraph")
        toLogicMod = importlib.import_module(f"Benchmark.{self.dataset_name}.toLogic")
        LogicConverter = getattr(toLogicMod, "toLogic")
    

        # split the data into train and test
        if split_data:
            self.benchmark.split_relational_data(self.relational_path, 
                                             self.dataset_name,
                                             class_name=self.target,
                                             problem_id=self.problem_id)

        self.graph_converter_train = GraphConverter(relational_path = self.relational_path_train,
                                             dataset_name = self.dataset_name,
                                             dataset_problem_key = self.problem_id,
                                             target = self.target)
        self.graph_converter_test = GraphConverter(relational_path = self.relational_path_test,
                                             dataset_name = self.dataset_name,
                                             dataset_problem_key = self.problem_id,
                                             target = self.target)
        self.logic_converter_train = LogicConverter(dataset_name = self.dataset_name,
                                              relational_path = self.relational_path_train,
                                              problem_key = self.problem_id)
        self.logic_converter_test = LogicConverter(dataset_name = self.dataset_name,
                                                relational_path = self.relational_path_test,
                                                problem_key = self.problem_id)

        self.IlpConverter = toILP

        self.ilp_settings = {
        "color":{
            "tilde":{
                "node_edge":[
                    "max_lookahead(4).\n",
                    "exhaustive_lookahead(2).\n",
                    "predict(color(+id,-C)).\n",
                    "warmode(node(+id,+node_id,-color)).\n",
                    "warmode(edge(+id,-node_id,-node_id)).\n",
                    "auto_lookahead(node(Id,Node_id,Color),[Node_id,Color]).\n",
                    "auto_lookahead(edge(Id,Node_id1,Node_id2),[Node_id1,Node_id2]).\n",
                ]
            },
            "aleph":{
                "node_edge":[
                    ":- aleph_set(clauselength,7).\n",
                    ":- modeb(*,node(+id,-color)).\n",
                    ":- modeb(*,edge(+id,-id)).\n",
                    ":- modeb(*,red(+color)).\n",
                    ":- modeb(*,green(+color)).\n",
                    ":- determination(color/1,node/3).\n",
                    ":- determination(color/1,edge/3).\n",
                    ":- determination(color/1,red/1).\n",
                    ":- determination(color/1,green/1).\n",
                ]
            }

        },
        "krk": {
            "tilde": {
                "node_only": [
                    f"predict({dataset_name}(+B,-C)).\n",
                    "rmode(white_king(+id,-node_id,-x,-y)).\n",
                    "rmode(white_rook(+id,-node_id,-x,-y)).\n",
                    "rmode(black_king(+id,-node_id,-x,-y)).\n",
                    "rmode(edge(+id,-node_id,-node_id)).\n",
                    # "rmode(same_x(+X1,#[0,1,2,3,4,5,6,7])).\n",
                    # "rmode(same_y(+Y1,#[0,1,2,3,4,5,6,7])).\n",
                    "typed_language(yes).\n",
                    f"type({dataset_name}(id,class)).\n",
                    "type(white_king(id,node_id,x,y)).\n",
                    "type(white_rook(id,node_id,x,y)).\n",
                    "type(black_king(id,node_id,x,y)).\n",
                    "type(edge(id,node_id,node_id)).\n",
                    # "type(same_x(x,x)).\n",
                    # "type(same_y(y,y)).\n",
                    "auto_lookahead(white_king(Id,Node_id,X,Y),[Node_id]).\n",
                    "auto_lookahead(white_rook(Id,Node_id,X,Y),[Node_id]).\n",
                    "auto_lookahead(black_king(Id,Node_id,X,Y),[Node_id]).\n",
                    "auto_lookahead(edge(Id,Node_id1,Node_id2),[Node_id1,Node_id2]).\n",
                    # "auto_lookahead(same_x(X1,X2),[X1,X2]).\n",
                    # "auto_lookahead(same_y(Y1,Y2),[Y1,Y2]).\n",
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
                    "warmode(distance(white_king(+id,+node_id,+x,+y),black_king(+id,+node_id,+x,+y),#d)).\n",
                    "warmode(one(#d)).\n",

                    # "typed_language(yes).\n",
                    # f"type({dataset_name}(id,class)).\n",
                    # "type(white_king(id,node_id,x,y)).\n",
                    # "type(white_rook(id,node_id,x,y)).\n",
                    # "type(black_king(id,node_id,x,y)).\n",
                    # "type(edge(id,node_id,node_id)).\n",
                    # "type(same_rank(id,node_id,node_id)).\n",
                    # "type(same_file(id,node_id,node_id)).\n",
                    # "type(same_x(x,x)).\n",
                    # "type(same_y(y,y)).\n",

                    # "rmode(same_x(+X1,#[0,1,2,3,4,5,6,7])).\n",
                    # "rmode(same_y(+Y1,#[0,1,2,3,4,5,6,7])).\n",

                    "auto_lookahead(white_king(Id,Node_id,X,Y),[Node_id,X,Y]).\n",
                    "auto_lookahead(white_rook(Id,Node_id,X,Y),[Node_id,X,Y]).\n",
                    "auto_lookahead(black_king(Id,Node_id,X,Y),[Node_id,X,Y]).\n",
                    "auto_lookahead(edge(Id,Node_id1,Node_id2),[Node_id1,Node_id2]).\n",
                    "auto_lookahead(same_rank(Id,Node_id1,Node_id2),[Node_id1,Node_id2]).\n",
                    "auto_lookahead(same_file(Id,Node_id1,Node_id2),[Node_id1,Node_id2]).\n",
                    "auto_lookahead(distance(white_king(Id,Node_id,X,Y),black_king(Id,Node_id,X,Y),D),[D]).\n",
                    "auto_lookahead(one(D),[D]).\n",
                    # "auto_lookahead(same_row(Y1,Y2),[Y1,Y2]).\n",
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
                    # "rmode(same_x(+X1,#[0,1,2,3,4,5,6,7])).\n",
                    # "rmode(same_y(+Y1,#[0,1,2,3,4,5,6,7])).\n",
                    "typed_language(yes).\n",
                    # "type(same_x(x,x)).\n",
                    # "type(same_y(y,y)).\n",
                    "auto_lookahead(white_king(Id,Node_id,X,Y),[Node_id,X,Y]).\n",
                    "auto_lookahead(white_rook(Id,Node_id,X,Y),[Node_id,X,Y]).\n",
                    "auto_lookahead(black_king(Id,Node_id,X,Y),[Node_id,X,Y]).\n",
                    "auto_lookahead(edge(Id,Node_id1,Node_id2),[Node_id1,Node_id2]).\n",
                    "auto_lookahead(same_rank(Id,Node_id1,Node_id2),[Node_id1,Node_id2]).\n",
                    "auto_lookahead(same_file(Id,Node_id1,Node_id2),[Node_id1,Node_id2]).\n",
                    "auto_lookahead(instance(Id,white_king,Node_id),[Node_id]).\n",
                    "auto_lookahead(instance(Id,white_rook,Node_id),[Node_id]).\n",
                    "auto_lookahead(instance(Id,black_king,Node_id),[Node_id]).\n",
                    # "auto_lookahead(same_x(X1,X2),[X1,X2]).\n",
                    # "auto_lookahead(same_y(Y1,Y2),[Y1,Y2]).\n",
                ],
                "Klog": [
                    "predict(krk(+B,-C)).\n",
                    "warmode(white_king(+id,+node_id,#x,#y)).\n",
                    "warmode(white_rook(+id,+node_id,#x,#y)).\n",
                    "warmode(black_king(+id,+node_id,#x,#y)).\n",
                    "warmode(same_rank(+id,-node_id)).\n",
                    "warmode(same_file(+id,-node_id)).\n",
                    "warmode(edge(+id,-node_id,-node_id)).\n",
                    # "rmode(same_x(+X1,#[0,1,2,3,4,5,6,7])).\n",
                    # "rmode(same_y(+Y1,#[0,1,2,3,4,5,6,7])).\n",
                    "typed_language(yes).\n",
                    # "type(same_x(x,x)).\n",
                    # "type(same_y(y,y)).\n",
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
                    # ":- modeb(*,same_x(+x, #[0,1,2,3,4,5,6,7])).\n",
                    # ":- modeb(*,same_y(+y, #[0,1,2,3,4,5,6,7])).\n",       
                    f":- determination({dataset_name}/1,white_king/4).\n",
                    f":- determination({dataset_name}/1,white_rook/4).\n",
                    f":- determination({dataset_name}/1,black_king/4).\n",
                    f":- determination({dataset_name}/1,edge/3).\n",
                    # f":- determination({dataset_name}/1,same_x/2).\n",
                    # f":- determination({dataset_name}/1,same_y/2).\n",
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
                    ":- modeb(*,distance(+x,+y,+x,+y,-d)).\n",
                    ":- modeb(*,one(+d)).\n",
                    f":- determination({dataset_name}/1,white_king/4).\n",
                    f":- determination({dataset_name}/1,white_rook/4).\n",
                    f":- determination({dataset_name}/1,black_king/4).\n",
                    f":- determination({dataset_name}/1,edge/3).\n",
                    f":- determination({dataset_name}/1,same_rank/3).\n",
                    f":- determination({dataset_name}/1,same_file/3).\n",
                    f":- determination({dataset_name}/1,same_x/2).\n",
                    f":- determination({dataset_name}/1,same_y/2).\n",
                    f":- determination({dataset_name}/1,distance/5).\n",
                    f":- determination({dataset_name}/1,one/1).\n",
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
        },
        "bongard": {
            "tilde": {
                "node_only" : [
                    "predict(bongard(+B,-C)).\n",
                    "rmode(triangle(+id,-object)).\n",
                    "rmode(square(+id,-object)).\n",
                    "rmode(circle(+id,-object)).\n",
                    "rmode(in(+id,-object)).\n",
                    "rmode(edge(+id,-object,-object)).\n",
                    "typed_language(yes).\n",
                    "type(bongard(pic,class)).\n",
                    "type(triangle(pic,obj)).\n",
                    "type(square(pic,obj)).\n",
                    "type(circle(pic,obj)).\n",
                    "type(in(pic,obj)).\n",
                    "type(edge(pic,obj,obj)).\n",
                    "auto_lookahead(triangle(Id,Obj),[Obj]).\n",
                    "auto_lookahead(square(Id,Obj),[Obj]).\n",
                    "auto_lookahead(circle(Id,Obj),[Obj]).\n",
                    "auto_lookahead(in(Id,Obj),[Obj]).\n",
                    "auto_lookahead(edge(Id,Obj1,Obj2),[Obj1,Obj2]).\n",
                ],
                "node_edge": [
                    "predict(bongard(+B,-C)).\n",
                    "rmode(triangle(+P,+-S)).\n",
                    "rmode(square(+P,+-S)).\n",
                    "rmode(circle(+P,+-S)).\n",
                    "rmode(edge(+P,+S1,+-S2)).\n",
                    "typed_language(yes).\n",
                    "type(bongard(pic,class)).\n",
                    "type(triangle(pic,obj)).\n",
                    "type(square(pic,obj)).\n",
                    "type(circle(pic,obj)).\n",
                    "type(edge(pic,obj,obj)).\n",
                    "auto_lookahead(triangle(Id,S),[S]).\n",
                    "auto_lookahead(square(Id,S),[S]).\n",
                    "auto_lookahead(circle(Id,S),[S]).\n",
                    "auto_lookahead(edge(Id,S1,S2),[S1,S2]).\n",
                ],
                "edge_based": [
                    "predict(bongard(+B,-C)).\n",
                    "rmode(shape(+P,triangle,+-C)).\n",
                    "rmode(shape(+P,square,+-C)).\n",
                    "rmode(shape(+P,circle,+-C)).\n",
                    "rmode(in(+P,+S1,+-S2)).\n",
                    "rmode(instance(+P,+-C)).\n",
                    "typed_language(yes).\n",
                    "type(bongard(pic,class)).\n",
                    "type(in(pic,obj,obj)).\n",
                    "type(shape(pic,fact,obj)).\n",
                    "type(instance(pic,obj)).\n",
                    "auto_lookahead(shape(Id,T,C),[T,C]).\n",
                    "auto_lookahead(in(Id,S1,S2),[S1,S2]).\n",
                    "auto_lookahead(instance(Id,C),[C]).\n",
                ],
                "Klog": [
                    "predict(bongard(+id,-C)).\n",
                    "rmode(triangle(+id,-object)).\n",
                    "rmode(square(+id,-object)).\n",
                    "rmode(circle(+id,-object)).\n",
                    "rmode(edge(+id,-object,-object)).\n",
                    "rmode(in(+id,-object)).\n",
                    "typed_language(yes).\n",
                    "type(bongard(pic,class)).\n",
                    "type(triangle(pic,obj)).\n",
                    "type(square(pic,obj)).\n",
                    "type(circle(pic,obj)).\n",
                    "type(edge(pic,obj,obj)).\n",
                    "type(in(pic,obj)).\n",
                    "auto_lookahead(triangle(Id,Obj),[Obj]).\n",
                    "auto_lookahead(square(Id,Obj),[Obj]).\n",
                    "auto_lookahead(circle(Id,Obj),[Obj]).\n",
                    "auto_lookahead(edge(Id,Obj1,Obj2),[Obj1,Obj2]).\n",
                    "auto_lookahead(in(Id,Obj),[Obj]).\n",
                ]
            },
            "popper": {
                "node_only": [
                    "body_pred(square,2).\n",
                    "body_pred(circle,2).\n",
                    "body_pred(triangle,2).\n",
                    "body_pred(in,2).\n",
                    "body_pred(edge,3).\n",
                    "type(bongard,(id,)).\n",
                    "type(square,(id,object)).\n",
                    "type(circle,(id,object)).\n",
                    "type(triangle,(id,object)).\n",
                    "type(in,(id,object)).\n",
                    "type(edge,(id,object,object)).\n",
                ],
                "node_edge": [
                    "body_pred(square,2).\n",
                    "body_pred(circle,2).\n",
                    "body_pred(triangle,2).\n",
                    "body_pred(edge,3).\n",
                    "type(bongard,(id,)).\n",
                    "type(square,(id,object)).\n",
                    "type(circle,(id,object)).\n",
                    "type(triangle,(id,object)).\n",
                    "type(edge,(id,object,object)).\n",
                ],
                "edge_based": [
                    "body_pred(shape,3).\n",
                    "body_pred(in,3).\n",
                    "type(bongard,(id,)).\n",
                    "type(in,(id,object,object)).\n"
                    "type(shape,(id,fact,object)).\n"
                ],
                "Klog": [
                    "body_pred(square,2).\n",
                    "body_pred(circle,2).\n",
                    "body_pred(triangle,2).\n",
                    "body_pred(edge,3).\n",
                    "body_pred(in,2).\n",
                    "type(bongard,(id,)).\n",
                    "type(square,(id,object)).\n",
                    "type(circle,(id,object)).\n",
                    "type(triangle,(id,object)).\n",
                    "type(in,(id,object)).\n",
                    "type(edge,(id,object,object)).\n",
                ],
            },
            "aleph": {
                "node_only": [
                    ":- modeb(*,square(+id, -object)).\n",
                    ":- modeb(*,circle(+id, -object)).\n",
                    ":- modeb(*,triangle(+id, -object)).\n",
                    ":- modeb(*,in(+id, -object)).\n",
                    ":- modeb(*,edge(+id, -object, -object)).\n",
                    ":- determination(bongard/1,square/2).\n",
                    ":- determination(bongard/1,circle/2).\n",
                    ":- determination(bongard/1,triangle/2).\n",
                    ":- determination(bongard/1,in/2).\n",
                    ":- determination(bongard/1,edge/3).\n"
                ],
                "node_edge": [
                    ":- modeb(*,square(+id, -object)).\n",
                    ":- modeb(*,circle(+id, -object)).\n",
                    ":- modeb(*,triangle(+id, -object)).\n",
                    ":- modeb(*,edge(+id, -object, -object)).\n",
                    ":- determination(bongard/1,square/2).\n",
                    ":- determination(bongard/1,circle/2).\n",
                    ":- determination(bongard/1,triangle/2).\n",
                    ":- determination(bongard/1,edge/3).\n"
                ],
                "edge_based": [
                    ":- modeb(1,square(+constant)).\n",
                    ":- modeb(1,circle(+constant)).\n",
                    ":- modeb(1,triangle(+constant)).\n",
                    ":- modeb(*,shape(+id, -constant, -object)).\n",
                    ":- modeb(*,in(+id, -object, -object)).\n",
                    ":- modeb(*,instance(+id, -object)).\n",
                    ":- determination(bongard/1,shape/3).\n",
                    ":- determination(bongard/1,in/3).\n",
                    ":- determination(bongard/1,square/1).\n",
                    ":- determination(bongard/1,circle/1).\n",
                    ":- determination(bongard/1,triangle/1).\n",
                    ":- determination(bongard/1,instance/2).\n"
                ],
                "Klog":[
                    ":- modeb(*,square(+id, -object)).\n",
                    ":- modeb(*,circle(+id, -object)).\n",
                    ":- modeb(*,triangle(+id, -object)).\n",
                    ":- modeb(*,in(+id, -object)).\n",
                    ":- modeb(*,edge(+id, -object, -object)).\n",
                    ":- determination(bongard/1,square/2).\n",
                    ":- determination(bongard/1,circle/2).\n",
                    ":- determination(bongard/1,triangle/2).\n",
                    ":- determination(bongard/1,in/2).\n"
                    ":- determination(bongard/1,edge/3).\n"
                ]
            }
        },
        "train": {
            "tilde": {
                "node_only" :[
                    f"predict({dataset_name}(+B,-C)).\n",
                    "warmode(has_car(+id,+-car_id)).\n",
                    "warmode(has_load(+id,+-car_id,+-load_id)).\n",
                    "warmode(short(+id,+-car_id)).\n",
                    "warmode(long(+id,+-car_id)).\n",
                    "warmode(two_wheels(+id,+-car_id)).\n",
                    "warmode(three_wheels(+id,+-car_id)).\n",
                    "warmode(roof_open(+id,+-car_id)).\n",
                    "warmode(roof_closed(+id,+-car_id)).\n",
                    "warmode(zero_load(+id,+-load_id)).\n",
                    "warmode(one_load(+id,+-load_id)).\n",
                    "warmode(two_load(+id,+-load_id)).\n",
                    "warmode(three_load(+id,+-load_id)).\n",
                    "warmode(circle(+id,+-load_id)).\n",
                    "warmode(triangle(+id,+-load_id)).\n",
                    "warmode(edge(+id,+-object,+-object)).\n",
                ],
                "node_edge": [
                    f"predict({dataset_name}(+B,-C)).\n",
                    "warmode(has_car(+id,+-car_id)).\n",
                    "warmode(has_load(+id,+-car_id,+-load_id)).\n",
                    "warmode(short(+id,+-car_id)).\n",
                    "warmode(long(+id,+-car_id)).\n",
                    "warmode(two_wheels(+id,+-car_id)).\n",
                    "warmode(three_wheels(+id,+-car_id)).\n",
                    "warmode(roof_open(+id,+-car_id)).\n",
                    "warmode(roof_closed(+id,+-car_id)).\n",
                    "warmode(zero_load(+id,+-load_id)).\n",
                    "warmode(one_load(+id,+-load_id)).\n",
                    "warmode(two_load(+id,+-load_id)).\n",
                    "warmode(three_load(+id,+-load_id)).\n",
                    "warmode(circle(+id,+-load_id)).\n",
                    "warmode(triangle(+id,+-load_id)).\n",
                ],
                "edge_based": [
                    f"predict({dataset_name}(+B,-C)).\n",
                    "warmode(has_car(+id,+-node_id,+-node_id)).\n",
                    "warmode(has_load(+id,+-node_id,+-node_id)).\n",
                    "warmode(edge(+id,+-node_id,+-object)).\n",
                    "warmode(instance(+id,+-node_id)).\n",
                    "warmode(train(+object)).\n",
                    "warmode(car(+object)).\n",
                    "warmode(load(+object)).\n",
                    "warmode(short(+object)).\n",
                    "warmode(long(+object)).\n",
                    "warmode(two_wheels(+object)).\n",
                    "warmode(three_wheels(+object)).\n",
                    "warmode(roof_open(+object)).\n",
                    "warmode(roof_closed(+object)).\n",
                    "warmode(zero_load(+object)).\n",
                    "warmode(one_load(+object)).\n",
                    "warmode(two_load(+object)).\n",
                    "warmode(three_load(+object)).\n",
                    "warmode(circle(+object)).\n",
                    "warmode(triangle(+object)).\n",
                ],
                "Klog": [
                    f"predict({dataset_name}(+B,-C)).\n",
                    "warmode(has_car(+id,+-car_id)).\n",
                    "warmode(has_load(+id,+-load_id)).\n",
                    "warmode(short(+id,+-car_id)).\n",
                    "warmode(long(+id,+-car_id)).\n",
                    "warmode(two_wheels(+id,+-car_id)).\n",
                    "warmode(three_wheels(+id,+-car_id)).\n",
                    "warmode(roof_open(+id,+-car_id)).\n",
                    "warmode(roof_closed(+id,+-car_id)).\n",
                    "warmode(zero_load(+id,+-load_id)).\n",
                    "warmode(one_load(+id,+-load_id)).\n",
                    "warmode(two_load(+id,+-load_id)).\n",
                    "warmode(three_load(+id,+-load_id)).\n",
                    "warmode(circle(+id,+-load_id)).\n",
                    "warmode(triangle(+id,+-load_id)).\n",
                    "warmode(edge(+id,+-object,+-object)).\n",
                ],
            },
            "aleph":{
                "node_only": [
                    ":- modeb(*,short(+id, -car_id)).\n",
                    ":- modeb(*,long(+id, -car_id)).\n",
                    ":- modeb(*,two_wheels(+id, -car_id)).\n",
                    ":- modeb(*,three_wheels(+id, -car_id)).\n",
                    ":- modeb(*,roof_open(+id, -car_id)).\n",
                    ":- modeb(*,roof_closed(+id, -car_id)).\n",
                    ":- modeb(*,zero_load(+id, -load_id)).\n",
                    ":- modeb(*,one_load(+id, -load_id)).\n",
                    ":- modeb(*,two_load(+id, -load_id)).\n",
                    ":- modeb(*,three_load(+id, -load_id)).\n",
                    ":- modeb(*,circle(+id, -load_id)).\n",
                    ":- modeb(*,triangle(+id, -load_id)).\n",
                    ":- modeb(*,edge(+puzzle, -object, -object)).\n",
                    f":- determination({dataset_name}/1,short/2).\n",
                    f":- determination({dataset_name}/1,long/2).\n",
                    f":- determination({dataset_name}/1,two_wheels/2).\n",
                    f":- determination({dataset_name}/1,three_wheels/2).\n",
                    f":- determination({dataset_name}/1,roof_open/2).\n",
                    f":- determination({dataset_name}/1,roof_closed/2).\n",
                    f":- determination({dataset_name}/1,zero_load/2).\n",
                    f":- determination({dataset_name}/1,one_load/2).\n",
                    f":- determination({dataset_name}/1,two_load/2).\n",
                    f":- determination({dataset_name}/1,three_load).\n"
                    f":- determination({dataset_name}/1,circle/2).\n",
                    f":- determination({dataset_name}/1,triangle/2).\n",
                    f":- determination({dataset_name}/1,edge/3).\n",
                ],
                "node_edge": [
                    ":- modeb(*,has_car(+id, -car_id)).\n",
                    ":- modeb(*,has_load(+id, -car_id, -load_id)).\n",
                    ":- modeb(*,short(+id, -car_id)).\n",
                    ":- modeb(*,long(+id, -car_id)).\n",
                    ":- modeb(*,two_wheels(+id, -car_id)).\n",
                    ":- modeb(*,three_wheels(+id, -car_id)).\n",
                    ":- modeb(*,roof_open(+id, -car_id)).\n",
                    ":- modeb(*,roof_closed(+id, -car_id)).\n",
                    ":- modeb(*,zero_load(+id, -load_id)).\n",
                    ":- modeb(*,one_load(+id, -load_id)).\n",
                    ":- modeb(*,two_load(+id, -load_id)).\n",
                    ":- modeb(*,three_load(+id, -load_id)).\n",
                    ":- modeb(*,circle(+id, -load_id)).\n",
                    ":- modeb(*,triangle(+id, -load_id)).\n",
                    f":- determination({dataset_name}/1,has_car/2).\n",
                    f":- determination({dataset_name}/1,has_load/3).\n",
                    f":- determination({dataset_name}/1,short/2).\n",
                    f":- determination({dataset_name}/1,long/2).\n",
                    f":- determination({dataset_name}/1,two_wheels/2).\n",
                    f":- determination({dataset_name}/1,three_wheels/2).\n",
                    f":- determination({dataset_name}/1,roof_open/2).\n",
                    f":- determination({dataset_name}/1,roof_closed/2).\n",
                    f":- determination({dataset_name}/1,zero_load/2).\n",
                    f":- determination({dataset_name}/1,one_load/2).\n",
                    f":- determination({dataset_name}/1,two_load/2).\n",
                    f":- determination({dataset_name}/1,three_load).\n"
                    f":- determination({dataset_name}/1,circle/2).\n",
                    f":- determination({dataset_name}/1,triangle/2).\n",
                ],
                "edge_based": [
                    ":- modeb(*,has_car(+id,-node_id,-node_id)).\n",
                    ":- modeb(*,has_load(+id,-node_id,-node_id)).\n",
                    ":- modeb(*,edge(+id,-node_id,-object)).\n",
                    ":- modeb(*,instance(+id,-node_id)).\n",
                    ":- modeb(*,train(+object)).\n",
                    ":- modeb(*,car(+object)).\n",
                    ":- modeb(*,load(+object)).\n",
                    ":- modeb(*,short(+object)).\n",
                    ":- modeb(*,long(+object)).\n",
                    ":- modeb(*,two_wheels(+object)).\n",
                    ":- modeb(*,three_wheels(+object)).\n",
                    ":- modeb(*,roof_open(+object)).\n",
                    ":- modeb(*,roof_closed(+object)).\n",
                    ":- modeb(*,zero_load(+object)).\n",
                    ":- modeb(*,one_load(+object)).\n",
                    ":- modeb(*,two_load(+object)).\n",
                    ":- modeb(*,three_load(+object)).\n",
                    ":- modeb(*,circle(+object)).\n",
                    ":- modeb(*,triangle(+object)).\n",
                    f":- determination({dataset_name}/1,has_car/3).\n",
                    f":- determination({dataset_name}/1,has_load/3).\n",
                    f":- determination({dataset_name}/1,edge/3).\n",
                    f":- determination({dataset_name}/1,instance/2).\n",
                    f":- determination({dataset_name}/1,train/1).\n",
                    f":- determination({dataset_name}/1,car/1).\n",
                    f":- determination({dataset_name}/1,load/1).\n",
                    f":- determination({dataset_name}/1,short/1).\n",
                    f":- determination({dataset_name}/1,long/1).\n",
                    f":- determination({dataset_name}/1,two_wheels/1).\n",
                    f":- determination({dataset_name}/1,three_wheels/1).\n",
                    f":- determination({dataset_name}/1,roof_open/1).\n",
                    f":- determination({dataset_name}/1,roof_closed/1).\n",
                    f":- determination({dataset_name}/1,zero_load/1).\n",
                    f":- determination({dataset_name}/1,one_load/1).\n",
                    f":- determination({dataset_name}/1,two_load/1).\n",
                    f":- determination({dataset_name}/1,three_load/1).\n",
                    f":- determination({dataset_name}/1,circle/1).\n",
                    f":- determination({dataset_name}/1,triangle/1).\n",

                ],
                "Klog": [
                    ":- modeb(*,has_car(+id, -car_id)).\n",
                    ":- modeb(*,has_load(+id, -load_id)).\n",
                    ":- modeb(*,short(+id, -car_id)).\n",
                    ":- modeb(*,long(+id, -car_id)).\n",
                    ":- modeb(*,two_wheels(+id, -car_id)).\n",
                    ":- modeb(*,three_wheels(+id, -car_id)).\n",
                    ":- modeb(*,roof_open(+id, -car_id)).\n",
                    ":- modeb(*,roof_closed(+id, -car_id)).\n",
                    ":- modeb(*,zero_load(+id, -load_id)).\n",
                    ":- modeb(*,one_load(+id, -load_id)).\n",
                    ":- modeb(*,two_load(+id, -load_id)).\n",
                    ":- modeb(*,three_load(+id, -load_id)).\n",
                    ":- modeb(*,circle(+id, -load_id)).\n",
                    ":- modeb(*,triangle(+id, -load_id)).\n",
                    ":- modeb(*,edge(+id, -object, -object)).\n",
                    f":- determination({dataset_name}/1,has_car/2).\n",
                    f":- determination({dataset_name}/1,has_load/2).\n",
                    f":- determination({dataset_name}/1,short/2).\n",
                    f":- determination({dataset_name}/1,long/2).\n",
                    f":- determination({dataset_name}/1,two_wheels/2).\n",
                    f":- determination({dataset_name}/1,three_wheels/2).\n",
                    f":- determination({dataset_name}/1,roof_open/2).\n",
                    f":- determination({dataset_name}/1,roof_closed/2).\n",
                    f":- determination({dataset_name}/1,zero_load/2).\n",
                    f":- determination({dataset_name}/1,one_load/2).\n",
                    f":- determination({dataset_name}/1,two_load/2).\n",
                    f":- determination({dataset_name}/1,three_load).\n"
                    f":- determination({dataset_name}/1,circle/2).\n",
                    f":- determination({dataset_name}/1,triangle/2).\n",
                    f":- determination({dataset_name}/1,edge/3).\n",
                ]
            },
            "popper":{
                "node_only": [
                    "body_pred(short,2).\n",
                    "body_pred(long,2).\n",
                    "body_pred(two_wheels,2).\n",
                    "body_pred(three_wheels,2).\n",
                    "body_pred(roof_open,2).\n",
                    "body_pred(roof_closed,2).\n",
                    "body_pred(zero_load,2).\n",
                    "body_pred(one_load,2).\n",
                    "body_pred(two_load,2).\n",
                    "body_pred(three_load,2).\n",
                    "body_pred(circle,2).\n",
                    "body_pred(triangle,2).\n",
                    "body_pred(edge,3).\n",
                    "type(train,(id,)).\n",
                    "type(short,(id,car_id)).\n",
                    "type(long,(id,car_id)).\n",
                    "type(two_wheels,(id,car_id)).\n",
                    "type(three_wheels,(id,car_id)).\n",
                    "type(roof_open,(id,car_id)).\n",
                    "type(roof_closed,(id,car_id)).\n",
                    "type(zero_load,(id,load_id)).\n",
                    "type(one_load,(id,load_id)).\n",
                    "type(two_load,(id,load_id)).\n",
                    "type(three_load,(id,load_id)).\n",
                    "type(circle,(id,load_id)).\n",
                    "type(triangle,(id,load_id)).\n"
                ],
                "node_edge": [
                    "body_pred(has_car,2).\n",
                    "body_pred(has_load,3).\n",
                    "body_pred(short,2).\n",
                    "body_pred(long,2).\n",
                    "body_pred(two_wheels,2).\n",
                    "body_pred(three_wheels,2).\n",
                    "body_pred(roof_open,2).\n",
                    "body_pred(roof_closed,2).\n",
                    "body_pred(zero_load,2).\n",
                    "body_pred(one_load,2).\n",
                    "body_pred(two_load,2).\n",
                    "body_pred(three_load,2).\n",
                    "body_pred(circle,2).\n",
                    "body_pred(triangle,2).\n",
                    "type(train,(id,)).\n",
                    "type(has_car,(id,car_id)).\n",
                    "type(has_load,(id,car_id,load_id)).\n",
                    "type(short,(id,car_id)).\n",
                    "type(long,(id,car_id)).\n",
                    "type(two_wheels,(id,car_id)).\n",
                    "type(three_wheels,(id,car_id)).\n",
                    "type(roof_open,(id,car_id)).\n",
                    "type(roof_closed,(id,car_id)).\n",
                    "type(zero_load,(id,load_id)).\n",
                    "type(one_load,(id,load_id)).\n",
                    "type(two_load,(id,load_id)).\n",
                    "type(three_load,(id,load_id)).\n",
                    "type(circle,(id,load_id)).\n",
                    "type(triangle,(id,load_id)).\n",
                ],
                "edge_based": [
                    "body_pred(has_car,3).\n",
                    "body_pred(has_load,3).\n",
                    "body_pred(edge,3).\n",
                    "body_pred(instance,2).\n",
                    "body_pred(train,1).\n",
                    "body_pred(car,1).\n",
                    "body_pred(load,1).\n",
                    "body_pred(short,1).\n",
                    "body_pred(long,1).\n",
                    "body_pred(two_wheels,1).\n",
                    "body_pred(three_wheels,1).\n",
                    "body_pred(roof_open,1).\n",
                    "body_pred(roof_closed,1).\n",
                    "body_pred(zero_load,1).\n",
                    "body_pred(one_load,1).\n",
                    "body_pred(two_load,1).\n",
                    "body_pred(three_load,1).\n",
                    "body_pred(circle,1).\n",
                    "body_pred(triangle,1).\n",
                    "type(train,(id,)).\n",
                    "type(has_car,(id,node_id,node_id)).\n",
                    "type(has_load,(id,node_id,node_id)).\n",
                    "type(edge,(id,node_id,object)).\n",
                    "type(instance,(id,node_id)).\n",
                    "type(car,(object,)).\n",
                    "type(load,(object,)).\n",
                    "type(short,(object,)).\n",
                    "type(long,(object,)).\n",
                    "type(two_wheels,(object,)).\n",
                    "type(three_wheels,(object,)).\n",
                    "type(roof_open,(object,)).\n",
                    "type(roof_closed,(object,)).\n",
                    "type(zero_load,(object,)).\n",
                    "type(one_load,(object,)).\n",
                    "type(two_load,(object,)).\n",
                    "type(three_load,(object,)).\n",
                    "type(circle,(object,)).\n",
                    "type(triangle,(object,)).\n",
                ],
                "Klog": [
                    "body_pred(has_car,2).\n",
                    "body_pred(has_load,2).\n",
                    "body_pred(short,2).\n",
                    "body_pred(long,2).\n",
                    "body_pred(two_wheels,2).\n",
                    "body_pred(three_wheels,2).\n",
                    "body_pred(roof_open,2).\n",
                    "body_pred(roof_closed,2).\n",
                    "body_pred(zero_load,2).\n",
                    "body_pred(one_load,2).\n",
                    "body_pred(two_load,2).\n",
                    "body_pred(three_load,2).\n",
                    "body_pred(circle,2).\n",
                    "body_pred(triangle,2).\n",
                    "body_pred(edge,3).\n",
                    "type(train,(id,)).\n",
                    "type(has_car,(id,car_id)).\n",
                    "type(has_load,(id,load_id)).\n",
                    "type(short,(id,car_id)).\n",
                    "type(long,(id,car_id)).\n",
                    "type(two_wheels,(id,car_id)).\n",
                    "type(three_wheels,(id,car_id)).\n",
                    "type(roof_open,(id,car_id)).\n",
                    "type(roof_closed,(id,car_id)).\n",
                    "type(zero_load,(id,load_id)).\n",
                    "type(one_load,(id,load_id)).\n",
                    "type(two_load,(id,load_id)).\n",
                    "type(three_load,(id,load_id)).\n",
                    "type(circle,(id,load_id)).\n",
                    "type(triangle,(id,load_id)).\n"
                ]
            }
        },
        "mutag": {
            "tilde": {
                "node_edge": [
                    "max_lookahead(1).\n",
                    #"exhaustive_lookahead(1).\n",
                    "query_batch_size(50000).\n",
                    "predict(mutag(+B,-C)).\n",
                    "warmode(atom(+id,-node_id,-atom_type,-charge)).\n",
                    "warmode(bond(+id,-node_id,-node_id,-bond_type)).\n",
                    "warmode(nitro(+id,-node_id,-node_id)).\n",
                    "warmode(benzene(+id,+-node_id,+-node_id)).\n",
                    "warmode(eq_atom(+atom_type,-atom_type)).\n",
                    "warmode(eq_bond(+bond_type,-bond_type)).\n",
                    "warmode(eq_charge(+charge,-charge)).\n",
                    "warmode(lteq(+charge,-charge)).\n",
                    "warmode(gteq(+charge,-charge)).\n",
                    "auto_lookahead(atom(Id,Node,Atom,Charge),[Node,Atom,Charge]).\n",
                    "auto_lookahead(bond(Id,Node_id_1,Node_id_2,Bond_type),[Node_id_1,Node_id_2,Bond_type]).\n",
                    "auto_lookahead(nitro(Id,Node_id_1,Node_id_2),[Node_id_1,Node_id_2]).\n",
                    "auto_lookahead(benzene(Id,Node_id_1,Node_id_2),[Node_id_1,Node_id_2]).\n",
                ]
            },
            "aleph": {
                "node_edge": [
                    ":- aleph_set(clauselength,15).\n",
                    ":- modeb(*,atom(+id, -object,-atom_type, -charge)).\n",
                    ":- modeb(*,bond(+id, -object, -object,-bond_type)).\n",
                    ":- modeb(*,nitro(+id, -object, -object)).\n",
                    ":- modeb(*,benzene(+id, -object, -object)).\n",
                    ":- modeb(*,eq_atom(+atom_type, -atom_type)).\n",
                    ":- modeb(*,eq_bond(+bond_type, -bond_type)).\n",
                    ":- modeb(*,eq_charge(+charge, -charge)).\n",
                    ":- modeb(*,lteq(+charge, -charge)).\n",
                    ":- modeb(*,gteq(+charge, -charge)).\n",
                    ":- determination(mutag/1,atom/4).\n",
                    ":- determination(mutag/1,bond/4).\n",
                    ":- determination(mutag/1,nitro/3).\n",
                    ":- determination(mutag/1,benzene/3).\n",
                    ":- determination(mutag/1,eq_atom/2).\n",
                    ":- determination(mutag/1,eq_bond/2).\n",
                    ":- determination(mutag/1,eq_charge/2).\n",

                ]
            }
        },
        "nci":{
            "tilde": {
                "node_only": [
                    f"predict({dataset_name}(+A,+B,+C,-D)).\n",
                    "warmode(parent(+id,+-node_id,+-name)).\n",
                    "warmode(child(+id,+-node_id,+-name)).\n",
                    "warmode(person(+id,+-node_id,+-name)).\n",
                    "warmode(edge(+id,+-node_id,+-node_id)).\n",
                ],
                "node_edge": [
                    "exhaustive_lookahead(1).\n",
                    "query_batch_size(50000).\n",
                    "max_lookahead(1).\n",
                    f"predict({dataset_name}(+id,-B)).\n",
                    "rmode(atom(+id,+node_id,-atom_type)).\n",
                    "rmode(bond(+id,+-node_id_1,+-node_id_2,-bond_type)).\n",
                    "rmode(eq_atom(+X,#[o,n,c,s,cl,p,f,na,sn,pt,ni,zn,mn,br,y,nd,ti,eu,cu,tl,zr,hf,in,ga,k,si,i])).\n",
                    "rmode(eq_bond(+X,#[1,2,3])).\n",
                    "rmode(connected(+id,+node_id_1,+node_id_2)).\n",
                    "typed_language(yes).\n",
                    f"type({dataset_name}(id,class)).\n",
                    "type(atom(id,node_id,atom_type)).\n",
                    "type(bond(id,node_id,node_id,bond_type)).\n",
                    "type(eq_atom(atom_type,atom_type)).\n",
                    "type(eq_bond(bond_type,bond_type)).\n",
                    "type(connected(id,node_id,node_id)).\n",
                    "auto_lookahead(atom(Id,Node,Atom),[Node,Atom]).\n",
                    "auto_lookahead(bond(Id,Node_id_1,Node_id_2,Bond_type),[Node_id_1,Node_id_2,Bond_type]).\n",
                ],
                "edge_based": [
                    f"predict({dataset_name}(+A,+B,+C,-D)).\n",
                    "warmode(person(+id,+-node_id,+-name)).\n",
                    "warmode(parent(+id,+-node_id,+-name,+-name)).\n",
                    "warmode(same_gen(+id,+-node_id,+-name1,+-name2)).\n",
                    "warmode(edge(+id,+-node_id,+-node_id)).\n",
                    "warmode(instance(+id,+-node_id)).\n",
                ],
                "Klog": [
                    f"predict({dataset_name}(+A,+B,+C,-D)).\n",
                    "warmode(person(+id,+-node_id,+-name)).\n",
                    "warmode(parent(+id,+-node_id)).\n",
                    "warmode(same_gen(+id,+-node_id)).\n",
                    "warmode(edge(+id,+-node_id,+-node_id)).\n",
                ]
            },
            "aleph": {
                "node_only": [
                ],
                "node_edge": [
                    ":- aleph_set(clauselength,10).\n",
                    ":- modeb(*,atom(+id, -node_id, -atom_type)).\n",
                    ":- modeb(*,bond(+id, -node_id, -node_id, -bond_type)).\n",
                    ":- modeb(*,nitro(+id, -node_id, -node_id)).\n",
                    ":- modeb(*,benzene(+id, +node_id, +node_id)).\n",
                    ":- modeb(*,eq_atom(+atom_type, -atom_type)).\n",
                    ":- modeb(*,eq_bond(+bond_type, -bond_type)).\n",
                    ":- determination(nci/1,atom/3).\n",
                    ":- determination(nci/1,bond/4).\n",
                    ":- determination(nci/1,eq_atom/2).\n",
                    ":- determination(nci/1,eq_bond/2).\n",
                    
                ],
                "edge_based": [
                ],
                "Klog": [
                ]
            }
        },
        "financial":{
            "tilde":{
                "node_edge": [
                    "max_lookahead(1).\n",
                    "query_batch_size(50000).\n",
                    "predict(financial(+B,-C)).\n",
                    "typed_language(yes).\n",
                    "type(account(key,district_id,frequency,date)).\n",
                    "type(card(key,disp_id,card_type,date)).\n",
                    "type(client(key,client_id,birth,district_id)).\n",
                    "type(disp(key,disp_id,client_id,disp_type)).\n",
                    "type(district(district_id,district_name,region,inhabitants,mun1,mun2,mun3,mun4,cities,ratio,avgsal,unemploy95,unemploy96,enterpreneurs,crimes95,crimes96)).\n",
                    "type(loan(key,date,amount,duration,payments)).\n",
                    "type(order(key,bank_to,amount,symbol)).\n",
                    "type(trans(key,date,trans_type,operation,amount,balance,trans_char)).\n",
                    "type(eq_amount(amount,amount)).\n",
                    "type(eq_avgsal(avgsal,avgsal)).\n",
                    "type(eq_balance(balance,balance)).\n",
                    "type(eq_bank_to(bank_to,bank_to)).\n",
                    "type(eq_card_type(card_type,card_type)).\n",
                    "type(eq_cities(cities,cities)).\n",
                    "type(eq_crimes95(crimes95,crimes95)).\n",
                    "type(eq_crimes96(crimes96,crimes96)).\n",
                    "type(eq_disp_type(disp_type,disp_type)).\n",
                    "type(eq_district_name(district_name,district_name)).\n",
                    "type(eq_duration(duration,duration)).\n",
                    "type(eq_enterpreneurs(enterpreneurs,enterpreneurs)).\n",
                    "type(eq_frequency(frequency,frequency)).\n",
                    "type(eq_inhabitants(inhabitants,inhabitants)).\n",
                    "type(eq_mun1(mun1,mun1)).\n",
                    "type(eq_mun2(mun2,mun2)).\n",
                    "type(eq_mun3(mun3,mun3)).\n",
                    "type(eq_mun4(mun4,mun4)).\n",
                    "type(eq_operation(operation,operation)).\n",
                    "type(eq_payments(payments,payments)).\n",
                    "type(eq_ratio(ratio,ratio)).\n",
                    "type(eq_region(region,region)).\n",
                    "type(eq_symbol(symbol,symbol)).\n",
                    "type(eq_trans_char(trans_char,trans_char)).\n",
                    "type(eq_trans_type(trans_type,trans_type)).\n",
                    "type(eq_unemploy95(unemploy95,unemploy95)).\n",
                    "type(eq_unemploy96(unemploy96,unemploy96)).\n",

                    "rmode(account(+Key0,+-District_id1,-Frequency2,+-Date3)).\n",
                    "rmode(card(+Key0,+-Disp_id1,-Card_type2,+-Date3)).\n",
                    "rmode(client(+Key0,+-Client_id1,+-Birth2,+-District_id3)).\n",
                    "rmode(disp(+Key0,+-Disp_id1,+-Client_id2,-Disp_type3)).\n",
                    "rmode(district(+District_id0,-District_name1,-Region2,-Inhabitants3,-Mun14,-Mun25,-Mun36,-Mun47,-Cities8,-Ratio9,-Avgsal10,-Unemploy9511,-Unemploy9612,-Enterpreneurs13,-Crimes9514,-Crimes9615)).\n",
                    "rmode(loan(+Key0,+-Date1,-Amount2,-Duration3,-Payments4)).\n",
                    "rmode(order(+Key0,-Bank_to1,-Amount2,-Symbol3)).\n",
                    "rmode(trans(+Key0,Date1,-Trans_type2,-Operation3,-Amount4,-Balance5,-Trans_char6)).\n",
                    
                    "auto_lookahead(account(Key0,District_id1,Frequency2,Date3),[District_id1,Frequency2,Date3]).\n",
                    "auto_lookahead(card(Key0,Disp_id1,Card_type2,Date3),[Disp_id1,Card_type2,Date3]).\n",
                    "auto_lookahead(client(Key0,Client_id1,Birth2,District_id3),[Client_id1,Birth2,District_id3]).\n",
                    "auto_lookahead(disp(Key0,Disp_id1,Client_id2,Disp_type3),[Disp_id1,Client_id2,Disp_type3]).\n",
                    "auto_lookahead(district(District_id0,District_name1,Region2,Inhabitants3,Mun14,Mun25,Mun36,Mun47,Cities8,Ratio9,Avgsal10,Unemploy9511,Unemploy9612,Enterpreneurs13,Crimes9514,Crimes9615),[District_name1,Region2,Inhabitants3,Mun14,Mun25,Mun36,Mun47,Cities8,Ratio9,Avgsal10,Unemploy9511,Unemploy9612,Enterpreneurs13,Crimes9514,Crimes9615]).\n",
                    "auto_lookahead(loan(Key0,Date1,Amount2,Duration3,Payments4),[Date1,Amount2,Duration3,Payments4]).\n",
                    "auto_lookahead(order(Key0,Bank_to1,Amount2,Symbol3),[Bank_to1,Amount2,Symbol3]).\n",
                    "auto_lookahead(trans(Key0,Date1,Trans_type2,Operation3,Amount4,Balance5,Trans_char6), [Date1,Trans_type2,Operation3,Amount4,Balance5,Trans_char6]).\n",

                    "rmode(eq_amount(+X, #['a13=_inf<x<=38772','a13=155028<x<=+inf','a13=38772<x<=68952','a13=68952<x<=93288','a13=93288<x<=155028','a16=_inf<x<=1292','a16=1292<x<=3018_35','a16=3018_35<x<=4379_85','a16=4379_85<x<=6822_5','a16=6822_5<x<=+inf','a17=_inf<x<=132_25','a17=132_25<x<=1973','a17=13932_5<x<=+inf','a17=1973<x<=5613_6','a17=5613_6<x<=13932_5'])).\n",
                    "rmode(eq_avgsal(+X, #['a7=_inf<x<=8402_5','a7=8402_5<x<=8691_5','a7=8691_5<x<=8966_5','a7=8966_5<x<=9637','a7=9637<x<=+inf'])).\n",
                    "rmode(eq_balance(+X, #['a18=_inf<x<=25048_7','a18=25048_7<x<=35687_1','a18=35687_1<x<=47669_9','a18=47669_9<x<=66077_4','a18=66077_4<x<=+inf'])).\n",
                    "rmode(eq_bank_to(+X, #[ab,cd,ef,gh,ij,kl,mn,op,qr,st,uv,wx,yz])).\n",
                    "rmode(eq_card_type(+X, #[classic,gold,junior])).\n",
                    "rmode(eq_cities(+X, #['a5=_inf<x<=4_5','a5=4_5<x<=5_5','a5=5_5<x<=6_5','a5=6_5<x<=8_5','a5=8_5<x<=+inf'])).\n",
                    "rmode(eq_crimes95(+X, #['a11=_inf<x<=1847_5','a11=1847_5<x<=2646_5','a11=2646_5<x<=3694','a11=3694<x<=5089','a11=5089<x<=+inf'])).\n",
                    "rmode(eq_crimes96(+X, #['a12=_inf<x<=1906_5','a12=1906_5<x<=2758_5','a12=2758_5<x<=3635_5','a12=3635_5<x<=5024_5','a12=5024_5<x<=+inf'])).\n",
                    "rmode(eq_disp_type(+X, #[disponent,owner])).\n",
                    "rmode(eq_district_name(+X, #[benesov,beroun,blansko,breclav,brno_mesto,brno_venkov,bruntal,ceska_lipa,ceske_budejovice,cesky_krumlov,cheb,chomutov,chrudim,decin,frydek_mistek,havlickuv_brod,hl_m_praha,hodonin,hradec_kralove,jablonec_n_nisou,jesenik,jicin,jindrichuv_hradec,karlovy_vary,karvina,kladno,kolin,kromeriz,kutna_hora,litomerice,louny,melnik,most,nachod,novy_jicin,nymburk,olomouc,opava,ostrava_mesto,pardubice,pelhrimov,pisek,plzen_jih,plzen_mesto,plzen_sever,prachatice,praha_vychod,praha_zapad,prerov,pribram,prostejov,rakovnik,rokycany,rychnov_nad_kneznou,semily,sokolov,strakonice,sumperk,svitavy,tabor,tachov,teplice,trebic,trutnov,uherske_hradiste,usti_nad_labem,usti_nad_orlici,vsetin,vyskov,zdar_nad_sazavou,zlin,znojmo])).\n",
                    "rmode(eq_duration(+X, #['a14=_inf<x<=18','a14=18<x<=30','a14=30<x<=42','a14=42<x<=54','a14=54<x<=+inf'])).\n",
                    "rmode(eq_enterpreneurs(+X, #['a10=_inf<x<=101','a10=101<x<=109_5','a10=109_5<x<=118_5','a10=118_5<x<=130_5','a10=130_5<x<=+inf'])).\n",
                    "rmode(eq_frequency(+X, #[poplatek_mesicne,poplatek_po_obratu,poplatek_tydne])).\n",
                    "rmode(eq_inhabitants(+X, #['a0=_inf<x<=76801','a0=119272<x<=159134','a0=159134<x<=+inf','a0=76801<x<=99258','a0=99258<x<=119272'])).\n",
                    "rmode(eq_mun1(+X, #['a1=_inf<x<=16','a1=16<x<=34_5','a1=34_5<x<=57','a1=57<x<=74_5','a1=74_5<x<=+inf'])).\n",
                    "rmode(eq_mun2(+X, #['a2=_inf<x<=13_5','a2=13_5<x<=21_5','a2=21_5<x<=27_5','a2=27_5<x<=35_5','a2=35_5<x<=+inf'])).\n",
                    "rmode(eq_mun3(+X, #['a3=_inf<x<=3_5','a3=3_5<x<=4_5','a3=4_5<x<=6_5','a3=6_5<x<=9_5','a3=9_5<x<=+inf'])).\n",
                    "rmode(eq_mun4(+X, #['a4=_inf<x<=0_5','a4=0_5<x<=1_5','a4=1_5<x<=2_5','a4=2_5<x<=3_5','a4=3_5<x<=+inf'])).\n",
                    "rmode(eq_operation(+X, #[prevod_na_ucet,prevod_z_uctu,vklad,vyber,vyber_kartou,none])).\n",
                    "rmode(eq_payments(+X, #['a15=_inf<x<=2163','a15=2163<x<=3537_5','a15=3537_5<x<=4974_5','a15=4974_5<x<=6717','a15=6717<x<=+inf'])).\n",
                    "rmode(eq_ratio(+X, #['a6=_inf<x<=49_45','a6=49_45<x<=56','a6=56<x<=62_45','a6=62_45<x<=80_25','a6=80_25<x<=+inf'])).\n",
                    "rmode(eq_region(+X, #[prague,central_bohemia,east_bohemia,north_bohemia,north_moravia,south_bohemia,south_moravia,west_bohemia])).\n",
                    "rmode(eq_symbol(+X, #[pojistne,sipo,uver,none])).\n",
                    "rmode(eq_trans_char(+X, #[pojistne,sankc_urok,sipo,sluzby,urok,uver,none])).\n",
                    "rmode(eq_trans_type(+X, #[prijem,vyber,vydaj])).\n",
                    "rmode(eq_unemploy95(+X, #['a8=_inf<x<=1_645','a8=1_645<x<=2_585','a8=2_585<x<=3_355','a8=3_355<x<=4_71','a8=4_71<x<=+inf'])).\n",
                    "rmode(eq_unemploy96(+X, #['a9=_inf<x<=2_14','a9=2_14<x<=3_24','a9=3_24<x<=4_07','a9=4_07<x<=5_505','a9=5_505<x<=+inf'])).\n",
                ]
            }
        },
        "cancer" :{
            "tilde": {
                "node_edge": [
                    "exhaustive_lookahead(1).\n",
                    "max_lookahead(3).\n",
                    "query_batch_size(50000).\n",
                    "predict(cancer(+B,-C)).\n",
                    "typed_language(yes).\n",
                    "type(atom(id,node_id,atom_type,charge)).\n",
                    "type(bond(id,node_id,node_id,bond_type)).\n",
                    "type(eq_atomtype(atom_type,atom_type)).\n",
                    "type(eq_charge(charge,charge)).\n",
                    "type(eq_bondtype(bond_type,bond_type)).\n",
                    "rmode(atom(+drug_id,+node_id,-atom_type,-charge)).\n",
                    "rmode(bond(+drug_id,+node_id_1,+node_id_2,-bond_type)).\n",
                    "auto_lookahead(atom(Id,Node,Atom,Charge),[Node,Atom,Charge]).\n",
                    "auto_lookahead(bond(Id,Node_id_1,Node_id_2,Bond_type),[Node_id_1,Node_id_2,Bond_type]).\n",
                    "auto_lookahead(eq_atomtype(Atom_type,Atom_type),[Atom_type]).\n",
                    "auto_lookahead(eq_charge(Charge,Charge),[Charge]).\n",
                    "auto_lookahead(eq_bondtype(Bond_type,Bond_type),[Bond_type]).\n",
                    "rmode(eq_atomtype(+X, #[1,10,101,102,113,115,120,121,129,134,14,15,16,17,19,191,192,193,2,21,22,232,26,27,29,3,31,32,33,34,35,36,37,38,40,41,42,45,49,499,50,51,52,53,60,61,62,70,72,74,75,76,77,78,79,8,81,83,84,85,87,92,93,94,95,96])).\n",
                    "rmode(eq_charge(+X, #['a0=_0_0175<x<=0_0615','a0=_0_1355<x<=_0_0175','a0=_inf<x<=_0_1355','a0=0_0615<x<=0_1375','a0=0_1375<x<=+inf'])).\n",
                    "rmode(eq_bondtype(+X, #[1,2,3,7])).\n",
                ]
            },
            "aleph": {
                "node_edge": [
                    ":- aleph_set(clauselength,15).\n",
                    ":- modeb(*,atom(+id, -node_id, -atom_type, -charge)).\n",
                    ":- modeb(*,bond(+id, -node_id, -node_id, -bond_type)).\n",
                    ":- modeb(*,eq_atomtype(+atom_type, -atom_type)).\n",
                    ":- modeb(*,eq_charge(+charge, -charge)).\n",
                    ":- modeb(*,eq_bondtype(+bond_type, -bond_type)).\n",
                    ":- modeb(*,cycle3(+id, -node_id, -node_id, -node_id)).\n",
                    ":- modeb(*,cycle4(+id, -node_id, -node_id, -node_id, -node_id)).\n",
                    ":- determination(cancer/1,atom/4).\n",
                    ":- determination(cancer/1,bond/4).\n",
                    ":- determination(cancer/1,eq_atomtype/2).\n",
                    ":- determination(cancer/1,eq_charge/2).\n",
                    ":- determination(cancer/1,eq_bondtype/2).\n",
                    ":- determination(cancer/1,cycle3/4).\n",
                    ":- determination(cancer/1,cycle4/5).\n",
                ]
            }
        },
        "cyclic": {
            "tilde" :{
                "node_only": [
                    "max_lookahead(1).\n",
                    "predict(cyclic(+B,-C)).\n",
                    "warmode(edge(+id_int,-id_int)).\n",
                    "warmode(node(+id,+-color)).\n",
                    "warmode(green(+color)).\n",
                    "warmode(red(+color)).\n",
                    "auto_lookahead(edge(Node_id_1,Node_id_2),[Node_id_1,Node_id_2]).\n",
                    "auto_lookahead(node(Node_id,Color),[Node_id,Color]).\n",
                ],
                "node_edge": [
                    "max_lookahead(1).\n",
                    "predict(cyclic(+B,-C)).\n",
                    "warmode(edge(+id,-id)).\n",
                    "warmode(node(+id,+-color)).\n",
                    "warmode(green(+color)).\n",
                    "warmode(red(+color)).\n",
                    "auto_lookahead(edge(Node_id_1,Node_id_2),[Node_id_1,Node_id_2]).\n",
                    "auto_lookahead(node(Node_id,Color),[Node_id,Color]).\n",
                ],
                "edge_based": [
                    "max_lookahead(1).\n",
                    "predict(cyclic(+B,-C)).\n",
                    "warmode(edge(+id,-id)).\n",
                    "warmode(node(+id,+-color)).\n",
                    "warmode(green(+color)).\n",
                    "warmode(red(+color)).\n",
                    "warmode(instance(+id)).\n",
                    "auto_lookahead(edge(Node_id_1,Node_id_2),[Node_id_1,Node_id_2]).\n",
                    "auto_lookahead(node(Node_id,Color),[Node_id,Color]).\n",
                    "auto_lookahead(instance(Id),[Id]).\n",
                ],
                "Klog": [
                    "max_lookahead(1).\n",
                    "predict(cyclic(+B,-C)).\n",
                    "warmode(edge(+id,-id)).\n",
                    "warmode(node(+id,+-color)).\n",
                    "warmode(green(+color)).\n",
                    "warmode(red(+color)).\n",
                    "warmode(klog_edge(+id)).\n",
                    "auto_lookahead(edge(Node_id_1,Node_id_2),[Node_id_1,Node_id_2]).\n",
                    "auto_lookahead(node(Node_id,Color),[Node_id,Color]).\n",
                    "auto_lookahead(klog_edge(Id),[Id]).\n",
                ]   
            },
            "aleph":{
                "node_only": [
                    ":- modeb(*,edge(+id_int,-id_int)).\n",
                    ":- modeb(*,node(+id,-color)).\n",
                    ":- modeb(*,green(+color)).\n",
                    ":- modeb(*,red(+color)).\n",
                    ":- determination(cyclic/1,edge/2).\n",
                    ":- determination(cyclic/1,node/2).\n",
                    ":- determination(cyclic/1,green/1).\n",
                    ":- determination(cyclic/1,red/1).\n",
                ],
                "node_edge": [
                    ":- aleph_set(clauselength,15).\n",
                    ":- modeb(*,edge(+id,-id)).\n",
                    ":- modeb(*,node(+id,-color)).\n",
                    ":- modeb(*,green(+color)).\n",
                    ":- modeb(*,red(+color)).\n",
                    ":- determination(cyclic/1,edge/2).\n",
                    ":- determination(cyclic/1,node/2).\n",
                    ":- determination(cyclic/1,green/1).\n",
                    ":- determination(cyclic/1,red/1).\n",
                ],
                "edge_based": [
                    ":- modeb(*,edge(+id,-id)).\n",
                    ":- modeb(*,node(+id,-color)).\n",
                    ":- modeb(*,green(+color)).\n",
                    ":- modeb(*,red(+color)).\n",
                    ":- modeb(*,instance(+id)).\n",
                    ":- determination(cyclic/1,edge/2).\n",
                    ":- determination(cyclic/1,node/2).\n",
                    ":- determination(cyclic/1,green/1).\n",
                    ":- determination(cyclic/1,red/1).\n",
                    ":- determination(cyclic/1,instance/1).\n",
                ],
                "Klog": [
                    ":- modeb(*,edge(+id,-id)).\n",
                    ":- modeb(*,node(+id,-color)).\n",
                    ":- modeb(*,green(+color)).\n",
                    ":- modeb(*,red(+color)).\n",
                    ":- modeb(*,klog_edge(+id)).\n",
                    ":- determination(cyclic/1,edge/2).\n",
                    ":- determination(cyclic/1,node/2).\n",
                    ":- determination(cyclic/1,green/1).\n",
                    ":- determination(cyclic/1,red/1).\n",
                    ":- determination(cyclic/1,klog_edge/1).\n",
                ]
            },
            "popper":{
                "node_only": [
                    "body_pred(edge,2).\n",
                    "body_pred(node,2).\n",
                    "body_pred(green,1).\n",
                    "body_pred(red,1).\n",
                    "type(cyclic,(id,)).\n",
                    "type(edge,(id_int,id_int)).\n",
                    "type(node,(id,color)).\n",
                    "type(green,(color,)).\n",
                    "type(red,(color,)).\n",
                    "direction(edge,(in,out)).\n",
                    "direction(node,(in,out)).\n",
                    "direction(green,(out,)).\n",
                    "direction(red,(out,)).\n",
                ],
                "node_edge": [
                    "body_pred(edge,2).\n",
                    "body_pred(node,2).\n",
                    "body_pred(green,1).\n",
                    "body_pred(red,1).\n",
                    "type(cyclic,(id,)).\n",
                    "type(node,(id,color)).\n",
                    "type(edge,(id,id)).\n",
                ],
                "edge_based": [
                    "body_pred(edge,2).\n",
                    "body_pred(node,2).\n",
                    "body_pred(green,1).\n",
                    "body_pred(red,1).\n",
                    "body_pred(instance,1).\n",
                    "type(cyclic,(id,)).\n",
                    "type(node,(id,color)).\n",
                    "type(edge,(id,id)).\n",
                    "type(instance,(id,)).\n",
                    "direction(cyclic,(in,)).\n",
                    "direction(node,(in,out)).\n",
                    "direction(edge,(in,out)).\n",
                    "direction(green,(out,)).\n",
                    "direction(red,(out,)).\n",
                    "direction(instance,(out,)).\n",
                ],
                "Klog": [
                    "body_pred(edge,2).\n",
                    "body_pred(node,2).\n",
                    "body_pred(green,1).\n",
                    "body_pred(red,1).\n",
                    "body_pred(klog_edge,1).\n",
                    "type(cyclic,(id,)).\n",
                    "type(node,(id,color)).\n",
                    "type(edge,(id,id)).\n",
                    "type(klog_edge,(id,)).\n",
                    "direction(cyclic,(in,)).\n",
                    "direction(node,(in,out)).\n",
                    "direction(edge,(in,out)).\n",
                    "direction(green,(out,)).\n",
                    "direction(red,(out,)).\n",
                    "direction(klog_edge,(out,)).\n",
                ]
            }
        },
        "ptc":{
            "tilde":{
            "node_edge": [
            ]
            },
            "aleph":{
                "node_edge": [
                    ":- modeb(*,atom(+id, -node_id, -atom_type)).\n",
                    ":- modeb(*,connected(+id,-node_id,-node_id,-bond_type)).\n",
                    ":- modeb(*,eq_atom(+atom_type, -atom_type)).\n",
                    ":- modeb(*,eq_bond(+bond_type, -bond_type)).\n",
                    ":- modeb(*,cycle3(+id, -node_id, -node_id, -node_id)).\n",
                    ":- modeb(*,cycle4(+id, -node_id, -node_id, -node_id, -node_id)).\n",
                    ":- modeb(*,cycle5(+id, -node_id, -node_id, -node_id, -node_id, -node_id)).\n",
                    ":- determination(ptc/1,atom/3).\n",
                    ":- determination(ptc/1,connected/4).\n"
                    ":- determination(ptc/1,eq_atom/2).\n",
                    ":- determination(ptc/1,eq_bond/2).\n"
                    ":- determination(ptc/1,cycle3/4).\n",
                    ":- determination(ptc/1,cycle4/5).\n",
                    ":- determination(ptc/1,cycle5/6).\n",
                ]
        }
    },
        "sameGen":{
            "tilde": {
                "node_only":[
                    f"predict({dataset_name}(+A,+B,+C,-D)).\n",
                    "warmode(parent(+id,+-node_id,+-name)).\n",
                    "warmode(child(+id,+-node_id,+-name)).\n",
                    "warmode(person(+id,+-node_id,+-name)).\n",
                    "warmode(edge(+id,+-node_id,+-node_id)).\n",
                ],
                "node_edge":[
                    f"predict({dataset_name}(+A,+B,+C,-D)).\n",
                    "warmode(person(+id,+-node_id,+-name)).\n",
                    "warmode(parent(+id,+-parent_name,+-name)).\n",
                    "warmode(same_gen(+id,+-name1,+-name2)).\n",
                    "typed_language(yes).\n",
                    f"type({dataset_name}(id,name,name,class)).\n",
                    "type(person,(id,node_id,name)).\n",
                    "type(parent,(id,name,name)).\n",
                    "type(same_gen,(id,name,name)).\n",
                    "auto_lookahead(person(Id,Node,Name),[Node,Name]).\n",
                    "auto_lookahead(parent(Id,Parent,Name),[Parent,Name]).\n",
                    "auto_lookahead(same_gen(Id,Name1,Name2),[Name1,Name2]).\n",
                ],
                "edge_based":[
                    f"predict({dataset_name}(+A,+B,+C,-D)).\n",
                    "warmode(person(+id,+-node_id,+-name)).\n",
                    "warmode(parent(+id,+-node_id,+-name,+-name)).\n",
                    "warmode(same_gen(+id,+-node_id,+-name1,+-name2)).\n",
                    "warmode(edge(+id,+-node_id,+-node_id)).\n",
                    "warmode(instance(+id,+-node_id)).\n",
                ],
                "Klog":[
                    f"predict({dataset_name}(+A,+B,+C,-D)).\n",
                    "warmode(person(+id,+-node_id,+-name)).\n",
                    "warmode(parent(+id,+-node_id)).\n",
                    "warmode(same_gen(+id,+-node_id)).\n",
                    "warmode(edge(+id,+-node_id,+-node_id)).\n",
                ]
                
            },
            "aleph": {
            },
            "popper": {
            }
        }
    }

        self.ilp_settings = self.ilp_settings[dataset_name]
        self.TILDE = True
        self.ALEPH = True
        self.POPPER = True
        self.GNN = True



    def run(self):
        
        # convert to the different representations
        self.convert_representations()

        # convert to ILP input formats
        self.convert_ilp()

        # run the ILP systems
        self.run_ilp()

        # run the GNN models
        self.run_gnn()
        
    def create_folders(self):
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_path = os.path.join(self.base_folder,self.dataset_name)

        # create representation folders
        for repr in self.representations:
            path = os.path.join(self.base_path, repr)
            if not os.path.exists(path):
                os.makedirs(path)
            logic_path = os.path.join(path, "logic")
            graph_path = os.path.join(path, "graph")
            if not os.path.exists(logic_path):
                os.makedirs(logic_path)
            for ilp in self.ilp_systems:
                ilp_path = os.path.join(logic_path, ilp)
                if not os.path.exists(ilp_path):
                    os.makedirs(ilp_path)
            if not os.path.exists(graph_path):
                os.makedirs(graph_path)
        # Add relational folder with train and test folders
        self.relational_path = os.path.join(self.base_path, "relational")
        relational_path = self.relational_path 
        if not os.path.exists(relational_path):
            os.makedirs(relational_path)
        self.relational_path_test = os.path.join(relational_path, "test")
        if not os.path.exists(self.relational_path_test):
            os.makedirs(self.relational_path_test)
        self.relational_path_train = os.path.join(relational_path, "train")
        if not os.path.exists(self.relational_path_train):
            os.makedirs(self.relational_path_train)
        # Add results folder
        results_path = os.path.join(self.base_path, "results")
        if not os.path.exists(results_path):
            os.makedirs(results_path)
    
    def convert_representations(self):
        for repr in self.representations:
            # build the graph representations
            string = f"self.graph_converter_train.{repr}()"
            test_string = f"self.graph_converter_test.{repr}()"
            graphs = eval(string)
            graphs_test = eval(test_string)

            # write the graphs to the graph directory
            torch.save(graphs, os.path.join(self.base_path, repr, "graph", "train.pt"))
            torch.save(graphs_test, os.path.join(self.base_path, repr, "graph", "test.pt"))

            # convert the graphs to logic
            if self.TILDE or self.ALEPH or self.POPPER:
                output_path = os.path.join(self.base_path, repr, "logic",self.dataset_name + ".kb")
                string = f"self.logic_converter_train.{repr}(graphs,'{output_path}')"
                eval(string)
                output_path_test = os.path.join(self.base_path, repr, "logic",self.dataset_name + "_test.kb")
                string = f"self.logic_converter_test.{repr}(graphs_test,'{output_path_test}')"
                eval(string)

                # remove the truth labels from the test file
                with open(output_path_test, "r") as file:
                    lines = file.readlines()
                new_test = []
                for line in lines:
                    if self.dataset_name not in line:
                        new_test.append(line)
                with open(output_path_test, "w") as file:
                    for line in new_test:
                        file.write(line)
    
    def convert_ilp(self):
        if self.TILDE or self.ALEPH or self.POPPER:
            for repr in self.representations:
                ilp_converter = self.IlpConverter(relational_path = self.relational_path,
                                                  logic_path = os.path.join(self.base_path, repr, "logic"),
                                                  dataset_name = self.dataset_name)
                logic_file_path = os.path.join(self.base_path, repr, "logic", self.dataset_name + ".kb")
                if self.POPPER:
                    ilp_converter.logicToPopper(logic_file_path = logic_file_path, label = self.dataset_name,bias_given = self.ilp_settings["popper"][repr])
                if self.TILDE:
                    ilp_converter.logicToTilde(logic_file_path = logic_file_path,givesettings = self.ilp_settings["tilde"][repr])
                if self.ALEPH:
                    ilp_converter.logicToAleph(logic_file_path = logic_file_path,label = self.dataset_name,given_settings = self.ilp_settings["aleph"][repr])

    def run_ilp(self):
        for repr in self.representations:
            results = pd.DataFrame()

            if self.TILDE:
                tilde = Tilde(dataset_name=self.dataset_name, relational_path=self.relational_path,target=self.target)
                tilde_results = tilde.run(tilde_input_path=os.path.join(self.base_path,repr))
                tilde_results['representation'] = repr
                results = pd.concat([results,tilde_results])
            if self.POPPER:
                popper = Popper(name=self.dataset_name,relational_path=self.relational_path,target=self.target)
                popper_results = popper.run(representation_path=os.path.join(self.base_path,repr))
                popper_results['representation'] = repr
                results = pd.concat([results,popper_results])
            if self.ALEPH:
                aleph = Aleph(name=self.dataset_name, relational_path=self.relational_path,target=self.target)
                aleph_results = aleph.run(representation_path=os.path.join(self.base_path,repr))
                aleph_results['representation'] = repr
                results = pd.concat([results,aleph_results])
            
            results.to_csv(os.path.join(self.base_path,"results",f"results_logic_{repr}.csv"),index=False)

        # merge the logic results of the representations
        if self.TILDE or self.POPPER or self.ALEPH:
            total_results = pd.DataFrame()
            for repr in self.representations:
                results = pd.read_csv(os.path.join(self.base_path,"results",f"results_logic_{repr}.csv"))
                os.remove(os.path.join(self.base_path,"results",f"results_logic_{repr}.csv"))
                total_results = pd.concat([total_results,results])

            if self.result_id is not None:
                total_results.to_csv(os.path.join(self.base_path,"results",f"results_logic_{self.result_id}.csv"),index=False)
            else:
                total_results.to_csv(os.path.join(self.base_path,"results","results_logic.csv"),index=False)

            # save the results per system as well
            for system in total_results['system'].unique():
                system_results = total_results[total_results['system'] == system]
                system_results.to_csv(os.path.join(self.base_path,"results",f"results_logic_{system}.csv"),index=False)
            
            print(total_results)
    

    def run_gnn(self):
        if self.GNN:
            with open("Benchmark/gnn_config.yaml") as file:
                config = yaml.safe_load(file)
            
            total_runs = len(config['models']) * len(config['layers']) * len(config['hidden_dims']) * len(self.representations) * config['repetitions']
            done_runs = 0
            total_gnn_data = pd.DataFrame()
            for model in config['models']:
                for layers in config['layers']:
                    for hidden_dims in config['hidden_dims']:
                        for representation in self.representations:
                            print(f"Run {done_runs}/{total_runs}")
                            graphs = torch.load(os.path.join(self.base_path,representation,"graph","train.pt"))
                            test_graphs = torch.load(os.path.join(self.base_path,representation,"graph","test.pt"))
                            result = self.benchmark.run_gnn(graphs,test_graphs,model,layers,hidden_dims,config)
                            result['representation'] = representation
                            total_gnn_data = pd.concat([total_gnn_data,result])
                            done_runs += config['repetitions']

            if self.result_id is not None:
                total_gnn_data.to_csv(os.path.join(self.base_path,"results",f"results_gnn_{self.result_id}.csv"),index=False)
            else:                   
                total_gnn_data.to_csv(os.path.join(self.base_path,"results","results_gnn.csv"),index=False)
            


if __name__ == "__main__":

    datasets = ["krk","bongard","train","mutag","nci","financial","cancer"]
    dataset_name = "bongard"
    representations = ["node_only","node_edge","edge_based","Klog"]
    representations = ["node_edge","edge_based","Klog"]
    representations = ["Klog"]
    
    target = "class"
    problem_id = "id"

    toGraphMod = importlib.import_module(f"Benchmark.{dataset_name}.toGraph")
    toGraph = getattr(toGraphMod, "toGraph")
    toLogicMod = importlib.import_module(f"Benchmark.{dataset_name}.toLogic")
    toLogic = getattr(toLogicMod, "toLogic")
    
    
    
    runner = Run(dataset_name,representations)
    runner.TILDE = False
    runner.ALEPH = True
    runner.POPPER = False
    runner.GNN = False
    runner.run()
