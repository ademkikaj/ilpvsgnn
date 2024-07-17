import pandas as pd
import numpy as np
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.data import HeteroData
from ..GraphConversion import GraphConversion
import json
import torch


class toGraph(GraphConversion):

    def __init__(self, relational_path, dataset_name, dataset_problem_key,target):
        super().__init__(relational_path, dataset_name, dataset_problem_key,target)
        self.encoders = self.load_encoders()
    
    def load_encoders(self):
        with open("Benchmark/financial/encoders.json") as f:
            encoders = json.load(f)
        return encoders

    def get_current_objects(self, index):
        current_objects = {}
        for key in self.kb.keys():
            if "id" in self.kb[key].columns:
                if index in self.kb[key]["id"].values:
                    current_objects[key] = self.kb[key][self.kb[key]["id"] == index]
                else:
                    current_objects[key] = pd.DataFrame()
            elif key == "district":
                client_district_ids = self.kb["client"][self.kb["client"]["id"] == index]["district_id"]
                account_district_ids = self.kb["account"][self.kb["account"]["id"] == index]["district_id"]
                all_district_ids = pd.concat([client_district_ids,account_district_ids])
                current_objects[key] = self.kb[key][self.kb[key]["district_id"].isin(all_district_ids)]
                #current_objects[key] = self.kb[key][self.kb[key]["district_id"].isin(self.kb["client"][self.kb["client"]["id"] == index]["district_id"] | 
                #                                                                     self.kb["account"][self.kb["account"]["id"] == index]["district_id"])] 
        return current_objects

    def create_node(self,kb,key,start_index,node_index,node_features,mapping=None,sec_key=None):
        for _,row in kb[key].iterrows():
            i = start_index
            t = torch.zeros(node_features.shape[1])
            for col in kb[key].columns:
                if "id" not in col.lower():
                    if col in self.encoders[key]:
                        t[i] = self.encoders[key][col][row[col]]
                        i += 1
                    else:
                        t[i] = row[col]
                        i += 1
                if mapping is not None:
                    if col == sec_key:
                        mapping[row[col]] = node_index
            node_features[node_index] = t
            node_index += 1
        return node_features, node_index
    

    def node_only(self):
        data_list = []
        node_tables = ["district","loan","order","trans","client","disp","account", "card", "client", "disp"]
        number_of_cols = {}
        total_node_features = 0
        for table in node_tables:
            df = self.kb[table]
            number_of_cols[table] = 0
            for col in df.columns:
                if "id" not in col.lower():
                    total_node_features += 1
                    number_of_cols[table] += 1

        for graph_index in range(self.num_of_graphs):

            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)

            total_nodes = sum([len(current_kb[key]) for key in node_tables])
            node_features =  torch.zeros((total_nodes, total_node_features))

            node_id = 0
            i = 0

            for table in node_tables:
                node_features,node_id = self.create_node(kb=current_kb,
                                                         key=table,
                                                         start_index=i,
                                                         node_index=node_id,
                                                         node_features=node_features)
                i += number_of_cols[table]
            
            edge_index = [[],[]]
            # create the fully connected graph
            for i in range(node_features.shape[0]):
                for j in range(node_features.shape[0]):
                    if i != j:
                        self.add_edges(edge_indices=edge_index,node_index1=i,node_index2=j)
            
            data_list.append(Data(x=node_features, edge_index=torch.tensor(edge_index),edge_attr=torch.ones(len(edge_index[0]),1,dtype=torch.float),y=torch.tensor([self.y[graph_index]])))

        return data_list
                            
    def node_edge(self):
        data_list = []

        number_of_cols = {}
        tables = ["account", "card", "client", "disp", "district", "loan", "order","trans"]
        node_tables = ["district","loan","order","trans","client","disp"]
        edge_tables = ["account", "card", "client", "disp"]
        total_node_features = 0
        total_edge_features = 0
        for table in tables:
            df = self.kb[table]
            number_of_cols[table] = 0
            for col in df.columns:
                if "id" not in col.lower():
                    if table in node_tables:
                        total_node_features += 1
                    elif table in edge_tables:
                        total_edge_features += 1
                    number_of_cols[table] += 1

        for graph_index in range(self.num_of_graphs):

            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)

            total_nodes = sum([len(current_kb[key]) for key in node_tables])
            node_features =  torch.zeros((total_nodes, total_node_features))

            
            node_id = 0
            i = 0
            # create the loan node: is the main node
            node_features,node_id = self.create_node(kb=current_kb,key="loan",node_index=node_id,start_index=i,node_features=node_features)
            i += number_of_cols["loan"]
            # create the order nodes
            node_features,node_id = self.create_node(kb=current_kb,key="order",node_index=node_id,start_index=i,node_features=node_features)
            i += number_of_cols["order"]
            # create the trans nodes
            node_features,node_id = self.create_node(kb=current_kb,key="trans",node_index=node_id,start_index=i,node_features=node_features)
            i += number_of_cols["trans"]
            # create the district nodes
            district_mapping = {}
            self.create_node(kb=current_kb,key="district",node_index=node_id,start_index=i,node_features=node_features,mapping=district_mapping,sec_key="district_id")
            i += number_of_cols["district"]
            # client is a table that implies a node and a relation between client and district
            client_mapping = {}
            self.create_node(kb=current_kb,key="client",node_index=node_id,start_index=i,node_features=node_features,mapping=client_mapping,sec_key="client_id")
            i += number_of_cols["client"]
            # disp is a table that implies a node and a relation between disp and client
            disp_mapping = {}
            self.create_node(kb=current_kb,key="disp",node_index=node_id,start_index=i,node_features=node_features,mapping=disp_mapping,sec_key="disp_id")
            i += number_of_cols["disp"]

            assert i == total_node_features
            # account and card are purely edges

            edge_index = [[],[]]
            edge_features = []

            j = 0
            # add account edges
            for _,row in current_kb["account"].iterrows():
                district_id_node = district_mapping[row["district_id"]]
                edge_feature = [0]*total_edge_features
                edge_feature[j] = self.encoders["account"]["frequency"][row["frequency"]]
                edge_feature[j+1] = row["date"]
                self.add_edges(edge_indices=edge_index,
                               node_index1=0,
                               node_index2=district_id_node,
                               edge_features=edge_features,
                               edge_feature=edge_feature)
            j += 2
            # add card edges
            for _,row in current_kb["card"].iterrows():
                disp_id_node = disp_mapping[row["disp_id"]]
                edge_feature = [0]*total_edge_features
                edge_feature[j] = self.encoders["card"]["card_type"][row["card_type"]]
                edge_feature[j+1] = row["date"]
                self.add_edges(edge_indices=edge_index,
                               node_index1=0,
                               node_index2=disp_id_node,
                               edge_features=edge_features,
                               edge_feature=edge_feature)
            j += 2
            # add client edges
            for _,row in current_kb["client"].iterrows():
                client_index = client_mapping[row["client_id"]]
                district_index = district_mapping[row["district_id"]]
                edge_feature = [0]*total_edge_features
                self.add_edges(edge_indices=edge_index,
                                 node_index1=client_index,
                                 node_index2=district_index,
                                 edge_features=edge_features,
                                 edge_feature=edge_feature)
            
            # add the disp edges
            for _,row in current_kb["disp"].iterrows():
                client_index = client_mapping[row["client_id"]]
                disp_index = disp_mapping[row["disp_id"]]
                edge_feature = [0]*total_edge_features
                self.add_edges(edge_indices=edge_index,
                               node_index1=client_index,
                               node_index2=disp_index,
                               edge_features=edge_features,
                               edge_feature=edge_feature)

            data_list.append(Data(x=node_features, edge_index=torch.tensor(edge_index), edge_attr=torch.tensor(edge_features), y=torch.tensor([self.y[graph_index]])))

        return data_list

    
    def edge_based(self):
        pass


    def Klog(self):
        data_list = []

        number_of_cols = {}
        tables = ["account", "card", "client", "disp", "district", "loan", "order","trans"]
        node_tables = ["district","loan","order","trans","client","disp"]
        edge_tables = ["account", "card", "client", "disp"]
        total_node_features = 0
        total_edge_features = 0
        for table in tables:
            df = self.kb[table]
            number_of_cols[table] = 0
            for col in df.columns:
                if "id" not in col.lower():
                    if table in node_tables:
                        total_node_features += 1
                    elif table in edge_tables:
                        total_edge_features += 1
                    number_of_cols[table] += 1

        for graph_index in range(self.num_of_graphs):

            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)

            total_nodes = sum([len(current_kb[key]) for key in tables]) + len(current_kb["client"]) + len(current_kb["disp"]) + len(current_kb["account"]) + len(current_kb["card"])
            num_node_features = total_node_features + total_edge_features + 2
            node_features =  torch.zeros((total_nodes, num_node_features))
            
            node_id = 0
            i = 0
            # create the loan node: is the main node
            node_features,node_id = self.create_node(kb=current_kb,key="loan",node_index=node_id,start_index=i,node_features=node_features)
            i += number_of_cols["loan"]
            # create the order nodes
            node_features,node_id = self.create_node(kb=current_kb,key="order",node_index=node_id,start_index=i,node_features=node_features)
            i += number_of_cols["order"]
            # create the trans nodes
            node_features,node_id = self.create_node(kb=current_kb,key="trans",node_index=node_id,start_index=i,node_features=node_features)
            i += number_of_cols["trans"]
            # create the district nodes
            district_mapping = {}
            self.create_node(kb=current_kb,key="district",node_index=node_id,start_index=i,node_features=node_features,mapping=district_mapping,sec_key="district_id")
            i += number_of_cols["district"]
            # client is a table that implies a node and a relation between client and district
            client_mapping = {}
            self.create_node(kb=current_kb,key="client",node_index=node_id,start_index=i,node_features=node_features,mapping=client_mapping,sec_key="client_id")
            i += number_of_cols["client"]
            # disp is a table that implies a node and a relation between disp and client
            disp_mapping = {}
            self.create_node(kb=current_kb,key="disp",node_index=node_id,start_index=i,node_features=node_features,mapping=disp_mapping,sec_key="disp_id")
            i += number_of_cols["disp"]

            assert i == total_node_features
            # account and card are purely edges

            edge_index = [[],[]]
            edge_features = []

            
            # add account edges
            for _,row in current_kb["account"].iterrows():
                # create the intermediate node
                node_features[node_id][i] = self.encoders["account"]["frequency"][row["frequency"]]
                node_features[node_id][i+1] = row["date"]
                # add edges to the main node
                self.add_edges(edge_indices=edge_index,node_index1=0,node_index2=node_id)

                district_id_node = district_mapping[row["district_id"]]
                
                self.add_edges(edge_indices=edge_index,
                               node_index1=node_id,
                               node_index2=district_id_node)
                node_id += 1
            i += 2

            # add card edges
            for _,row in current_kb["card"].iterrows():
                # create the intermediate node
                node_features[node_id][i] = self.encoders["card"]["card_type"][row["card_type"]]
                node_features[node_id][i+1] = row["date"]

                self.add_edges(edge_indices=edge_index,node_index1=0,node_index2=node_id)

                disp_id_node = disp_mapping[row["disp_id"]]
                
                self.add_edges(edge_indices=edge_index,node_index1=node_id,node_index2=disp_id_node)
                node_id += 1
            i += 2

            # add client edges
            for _,row in current_kb["client"].iterrows():
                # client node is already created
                # create the intermediate node
                node_features[node_id][i] = row["birth"]
                # connect client to intermediate node
                client_index = client_mapping[row["client_id"]]
                self.add_edges(edge_indices=edge_index,node_index1=client_index,node_index2=node_id)
                district_index = district_mapping[row["district_id"]]
                self.add_edges(edge_indices=edge_index,node_index1=client_index,node_index2=district_index)
                node_id += 1
            i += 1

            # add the disp edges
            for _,row in current_kb["disp"].iterrows():
                # disp node is already created
                # create the intermediate node
                node_features[node_id][i] = self.encoders["disp"]["disp_type"][row["disp_type"]]
                # connect client to intermediate node
                client_index = client_mapping[row["client_id"]]
                self.add_edges(edge_indices=edge_index,node_index1=client_index,node_index2=node_id)
                # connect other to intermediate node
                disp_index = disp_mapping[row["disp_id"]]
                self.add_edges(edge_indices=edge_index,node_index1=client_index,node_index2=disp_index)
                
                node_id += 1
            i += 1

            data_list.append(Data(x=node_features, edge_index=torch.tensor(edge_index), 
                                  edge_attr = torch.ones(len(edge_index[0]),1,dtype=torch.float), y=torch.tensor([self.y[graph_index]])))

        return data_list