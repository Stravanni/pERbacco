# imports
if (True):
    import json
    from argparse import ArgumentParser
    import networkx as nx
    import itertools
    #from pyscipopt import Model, quicksum, SCIP_PARAMSETTING
    import pandas as pd
    import numpy as np
    import torch
    #from torch.utils.data import Dataset
    from tqdm import tqdm
    import functools
    import ast
    import numpy as np
    import matplotlib.pyplot as plt
    from queue import PriorityQueue
    import math
    #import numba
    #from numba import cuda
    from multiprocessing import Process, Manager
    from sklearn.linear_model import LogisticRegression
    from IPython.display import display
    from brokenaxes import brokenaxes
    import re
    import matplotlib.ticker as ticker
    import statistics
    import copy
    import igraph as ig
    import leidenalg

    import os
    import sys
    from networkx.algorithms import community
    from networkx.algorithms.community import kernighan_lin_bisection
    import random
    from collections import Counter
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import pairwise_distances
    from scipy.sparse.csgraph import shortest_path
    import pickle
    from datetime import datetime
    import time
    import pyarrow.parquet as pq
    import fastparquet

    import concurrent.futures
    import time

def q_rec_k(x, k):
    if x < k:
        return 0
    else:
        n = x // k       # Integer division to find n
        x_prime = x % k  # Remainder for x'
        return n + q_rec_k(n + x_prime, k)

def r_rec_k(x, k):
    if x < k:
        return x
    else:
        n = x // k       # Integer division to find n
        x_prime = x % k  # Remainder for x'
        return r_rec_k(n + x_prime, k)

def read_graph(dname, synth_precision = "False"):

    g = pd.read_csv(f'datasets/{dname}/groundtruth.csv')
    g.columns.values[0] = 'id1'
    g.columns.values[1] = 'id2'


    if synth_precision == "False":
        path_graph = "similarity_graph/"+dname+".parquet"
    else:
        path_graph = "similarity_graph/synth_precision_"+str(synth_precision)+"/"+dname+".parquet"

    # If path_graph exists, then read the graph
    # otherwise compute the graph
    if os.path.exists(path_graph):
        df_graph = pd.read_parquet(path_graph)

        #print(df_graph.head(10))

        df_graph.rename(columns={"w": "weight"}, inplace=True)
        graph = nx.from_pandas_edgelist(df_graph, source='id1', target='id2', edge_attr='weight', create_using=nx.Graph())

        missing_nodes = set(g.values.flatten()) - graph.nodes()
        graph.add_nodes_from(missing_nodes)
    else:
        print("The graph doesn't exist!!!!")
        print(path_graph)
        STOPPO

    # Necessary step if the ids are not [1,k] but they are anonymized (random values in [1 ,10^10])

    set_nodes = set(graph.nodes())
    if len(set_nodes)  < max(set_nodes):
        #Different range!
        #print(len(set_nodes))
        #print(max(set_nodes))
        dict_conversion = dict(enumerate(set_nodes))
        dict_inv = {v: k for k, v in dict_conversion.items()}

        # Relabel the nodes in the graph
        graph = nx.relabel_nodes(graph, dict_inv)

        g['id1'] = g['id1'].map(dict_inv)
        g['id2'] = g['id2'].map(dict_inv)

    print(dname, "nodes, matches, edges", len(set_nodes), len(g), len(graph.edges()))

    return(g, graph)


def normalize_identifier(value):
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value).strip()
    if text == "":
        return ""
    try:
        return int(text)
    except ValueError:
        return text


def read_dataset_records(dname):
    path = f"datasets/{dname}/{dname}.csv"
    if not os.path.exists(path):
        return {}

    df_records = pd.read_csv(path, dtype=str).fillna("")
    if len(df_records.columns) == 0:
        return {}

    first_column = df_records.columns[0]
    if first_column != "id":
        df_records = df_records.rename(columns={first_column: "id"})

    records = {}
    for _, row in df_records.iterrows():
        record = {column: str(value).strip() for column, value in row.items()}
        identifier = normalize_identifier(record.get("id"))
        records[identifier] = record

    return records


ORACLE_STOPWORDS = {
    "the",
    "of",
    "and",
    "in",
    "on",
    "for",
    "to",
    "a",
    "an",
    "proceedings",
    "proc",
    "conference",
    "workshop",
    "journal",
    "international",
    "annual",
    "symposium",
}

def synthetic_dataset (number_entities, size_max, recall, precision, seed = 0):
    random.seed(seed)
    np.random.seed(seed)

    # Generate power-law integers that are the dimension of entities
    alpha = 2      # power-law exponent (typical values ~1.5–3)
    data = np.random.zipf(alpha, number_entities)
    data = data[data <= size_max]
    data = list(data)

    print("number of nodes:", sum(data))
    #print(sorted(data, reverse = True))

    ############################################################

    # create the graph G and the ground-truth g

    clique_sizes = data

    G = nx.disjoint_union_all(
        [nx.complete_graph(k) for k in clique_sizes]
    )


    # create the ground-truth g
    g = pd.DataFrame(index=range(len(G.edges)), columns=["id1","id2"])

    for i,edge in enumerate(list(G.edges())):
        x,y = edge
        g.loc[i] = [x,y]

    # add and remove edges for specific recall and precision

    number_match = len(G.edges())
    number_edge_to_remove = int(number_match * (1-recall))
    number_edge = number_match - number_edge_to_remove

    number_edges_to_add = int((number_edge - precision*number_edge)/precision)
    original_edges = list(G.edges())

    # add random weights in (0,1) for already existing edges
    for u, v in G.edges():
        G[u][v]['weight'] = round(random.random(),4)

    # add edges with Pareto distribution of weights
    #print(len(G.edges()))
    nodes = list(G.nodes())
    added = 0



    while added < number_edges_to_add:
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            #G[u][v]['weight'] = round(random.random(),4)
            G[u][v]["weight"] = round(np.random.pareto(30),4)
            added += 1

    #print(len(G.edges()))

    # remove edges
    edges_to_remove = random.sample(original_edges, number_edge_to_remove)
    G.remove_edges_from(edges_to_remove)
    #print(len(G.edges()))

    return (g, G)


class class_entity:
    def __init__(
        self,
        dname,
        graph,
        df_ground_truth,
        batch_size,
        alg_community,
        mu_benefit,
        lambda_w,
        oracle_backend="groundtruth",
        openai_model=None,
        prompt_mode="zero-shot",
    ):

        #graph_copy = copy.deepcopy(graph)
        self.dname = dname
        self.graph = nx.Graph(graph)
        self.nodes = len(self.graph.nodes())
        self.igraph = None
        self.df_ground_truth = df_ground_truth
        self.dict_ground_truth = self.create_dict_ground_truth()
        self.batch_size = batch_size
        self.alg_community = alg_community
        self.mu_benefit = mu_benefit
        self.lambda_w = lambda_w
        self.oracle_backend = oracle_backend
        self.openai_model = openai_model
        self.prompt_mode = prompt_mode
        self.record_index = read_dataset_records(dname)
        self.oracle_client = None
        self.last_query_stats = self.empty_query_stats()

        self.dict_entity = dict()
        self.dict_entity_belonging = dict()
        self.df_benefit = pd.DataFrame(columns=["entity1", "entity2", "benefit"])
        self.list_community = None
        self.dict_community = dict() 
        self.dict_community_inverted = dict()
        self.recall = None
        self.recall_community = None
        self.recall_community_max = None

        self.list_prob_match = list()
        self.list_prob_no_match = list()
        self.list_benefit_match = list()
        self.list_benefit_no_match = list()
        self.threshold_prob = 0

        self.temperature = 0
        self.with_community = None

        # NORMALIZATION OF EDGE WEIGHTS
        max_weight = max(data['weight'] for u, v, data in self.graph.edges(data=True) if 'weight' in data)

        for i, j, data in self.graph.edges(data=True):
            weight = data.get('weight')
            self.graph[i][j]['weight'] = weight/max_weight

        # TRANSORM THE WEIGHTS INTO INTEGERS
        for i, j, data in self.graph.edges(data=True):
            weight = data.get('weight')
            self.graph[i][j]['weight'] = round(weight,6) * 1000000
       
        self.max_weight = max(data['weight'] for _, _, data in self.graph.edges(data=True))

        self.dict_row_df_benefit = dict()
        self.max_index_df_benefit = 0
        
        #self.community = community
        self.number_heavy = None
        self.sum_heavy_comm = 0

        self.multigraph = nx.MultiGraph(self.graph)

        for u, v, data in self.multigraph.edges(data=True):
            data["max_weight"] = data["weight"]

        self.dist_matrix = None

        if self.oracle_backend == "openai":
            from oracle_LL import OpenAIEntityOracle

            self.oracle_client = OpenAIEntityOracle(
                model=self.openai_model,
                prompt_mode=self.prompt_mode,
            )

    def empty_query_stats(self):
        return {
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
            "llm_total_tokens": 0,
            "response_id": "",
        }

    def lookup_record(self, record_id):
        normalized = normalize_identifier(record_id)
        record = self.record_index.get(normalized)
        if record is not None:
            return record
        return {"id": str(record_id), "note": f"raw_id={record_id}"}

    def normalize_text_for_oracle(self, text):
        text = str(text or "").lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        tokens = [token for token in text.split() if token and token not in ORACLE_STOPWORDS]
        return tokens

    def extract_author_last_names(self, author_text):
        author_text = str(author_text or "").lower()
        author_text = re.sub(r"[^a-z,; ]+", " ", author_text)
        segments = [segment.strip() for segment in re.split(r"\band\b|;|,", author_text) if segment.strip()]
        surnames = []
        for segment in segments:
            pieces = [piece for piece in segment.split() if piece]
            if not pieces:
                continue
            surname = pieces[-1]
            if len(surname) > 1 and surname not in ORACLE_STOPWORDS:
                surnames.append(surname)
        return sorted(set(surnames))

    def normalize_numeric_signature(self, value):
        digits = re.findall(r"\d+", str(value or ""))
        return "-".join(digits[:4])

    def summarize_record_for_oracle(self, record):
        title_tokens = self.normalize_text_for_oracle(record.get("title", ""))
        venue_tokens = self.normalize_text_for_oracle(record.get("venue", ""))
        summary = {
            "id": str(record.get("id", "")),
            "title": record.get("title", ""),
            "author": record.get("author", ""),
            "year": record.get("year", ""),
            "venue": record.get("venue", ""),
            "pages": record.get("pages", ""),
            "publisher": record.get("publisher", ""),
            "editor": record.get("editor", ""),
            "institution": record.get("institution", ""),
            "note": record.get("note", ""),
            "normalized_title_tokens": title_tokens[:12],
            "author_last_names": self.extract_author_last_names(record.get("author", ""))[:8],
            "year_signature": self.normalize_numeric_signature(record.get("year", "")),
            "page_signature": self.normalize_numeric_signature(record.get("pages", "")),
            "venue_tokens": venue_tokens[:10],
        }
        return summary

    def build_oracle_entity_payload(self, entity_id):
        entity = self.dict_entity.get(entity_id)
        if entity is None:
            member_ids = [entity_id]
        else:
            member_ids = sorted(entity["set_entity"])

        records = []
        title_variants = set()
        author_last_names = set()
        year_values = set()
        page_signatures = set()
        venue_signatures = set()
        for member_id in member_ids[:5]:
            record = self.lookup_record(member_id).copy()
            record["id"] = str(record.get("id", member_id))
            summary = self.summarize_record_for_oracle(record)
            records.append(summary)
            title_variants.add(" ".join(summary["normalized_title_tokens"]))
            author_last_names.update(summary["author_last_names"])
            if summary["year_signature"]:
                year_values.add(summary["year_signature"])
            if summary["page_signature"]:
                page_signatures.add(summary["page_signature"])
            venue_signatures.add(" ".join(summary["venue_tokens"]))

        return {
            "entity_id": str(entity_id),
            "entity_size": len(member_ids),
            "normalized_title_variants": sorted(value for value in title_variants if value)[:5],
            "author_last_names_union": sorted(author_last_names)[:12],
            "year_signatures": sorted(year_values),
            "page_signatures": sorted(page_signatures),
            "venue_signatures": sorted(value for value in venue_signatures if value)[:5],
            "records": records,
        }

    def _resolve_query_matches(self, set_query):
        self.last_query_stats = self.empty_query_stats()

        if self.oracle_backend != "openai":
            return None

        if self.oracle_client is None:
            raise RuntimeError("OpenAI oracle backend selected, but the oracle client is not configured.")

        normalized_query = sorted(set(set_query))
        entities = [self.build_oracle_entity_payload(entity_id) for entity_id in normalized_query]
        id_lookup = {str(entity_id): entity_id for entity_id in normalized_query}

        response = self.oracle_client.resolve_batch(entities)
        self.last_query_stats = {
            **self.empty_query_stats(),
            **response["usage"],
            "response_id": response.get("response_id", ""),
        }

        match_pairs = set()
        for cluster in response["clusters"]:
            cluster_ids = [id_lookup[entity_id] for entity_id in cluster["entity_ids"]]
            for left, right in itertools.combinations(cluster_ids, 2):
                match_pairs.add(frozenset({left, right}))

        return match_pairs

    def entities_match(self, left, right, match_pairs=None):
        if self.oracle_backend == "openai":
            return frozenset({left, right}) in (match_pairs or set())
        return left in self.dict_ground_truth[right]

    def compute_predicted_pairs(self):
        total_predicted = 0
        for key in self.dict_entity.keys():
            nodes_key = len(self.dict_entity[key]["set_entity"])
            total_predicted += int(nodes_key * (nodes_key - 1) / 2)
        return total_predicted

    def compute_metrics(self):
        true_positive_pairs = 0
        predicted_pairs = 0
        for entity in self.dict_entity.values():
            members = sorted(entity["set_entity"])
            predicted_pairs += int(len(members) * (len(members) - 1) / 2)
            for left, right in itertools.combinations(members, 2):
                if left in self.dict_ground_truth[right]:
                    true_positive_pairs += 1

        recall = true_positive_pairs / len(self.df_ground_truth)
        if predicted_pairs == 0:
            precision = 1.0
        else:
            precision = true_positive_pairs / predicted_pairs

        return {
            "recall": recall,
            "precision": precision,
            "true_positive_pairs": true_positive_pairs,
            "predicted_pairs": predicted_pairs,
        }

    def build_estimation_batches(self, batch_size):
        if self.oracle_client is None:
            return {}

        nodes = sorted(self.graph.nodes())
        if len(nodes) == 0:
            return {}

        scored_nodes = []
        for node in nodes:
            payload = self.build_oracle_entity_payload(node)
            score = len(json.dumps(payload, ensure_ascii=True))
            scored_nodes.append((score, node))

        scored_nodes.sort(key=lambda item: item[0])
        positions = {
            "low": int(0.25 * (len(scored_nodes) - 1)),
            "avg": int(0.50 * (len(scored_nodes) - 1)),
            "high": int(0.75 * (len(scored_nodes) - 1)),
        }

        estimates = {}
        for label, index in positions.items():
            start = max(0, min(index - batch_size // 2, len(scored_nodes) - batch_size))
            selection = [node for _, node in scored_nodes[start : start + batch_size]]
            entities = [self.build_oracle_entity_payload(node) for node in selection]
            estimates[label] = self.oracle_client.estimate_batch_tokens(entities)

        return estimates


    # Compute the weight of a set (the sum of all weights)
    def compute_weight (self, set1):
        if len(set1) == 1:
            return(0)

        H = self.graph.subgraph(set1)
        weight = sum(data['weight'] for _, _, data in H.edges(data=True))
        return (weight)

    def compute_density (self, set1):
        if len(set1) == 1:
            return(0)

        H = self.graph.subgraph(set1)
        weight = sum(data['weight'] for _, _, data in H.edges(data=True))
        nodes_H = len(set1)
        denominator = (nodes_H*(nodes_H-1)/2)
        normalized_weight = (weight/self.max_weight)/denominator
        return (normalized_weight)
        

    # Compute the benefit between two entities
    def compute_benefit (self, u, v):

        edges = self.multigraph.get_edge_data(u,v)

        if edges:
            edge = edges[0]
            weight = edge["weight"]
            max_weight = edge["max_weight"]

            size_u = len(self.dict_entity[u]["set_entity"])
            size_v = len(self.dict_entity[v]["set_entity"])

            if self.mu_benefit == "brmax":
                prob = max_weight/self.max_weight
            if self.mu_benefit == "brmean":
                prob = (weight/self.max_weight)/(size_u * size_v)
        
            benefit = prob * size_u * size_v

            return (benefit, prob)
        else: # there are no edges between u and v
            return (0, 0)
 
    # Dict ground truth
    def create_dict_ground_truth (self):
        
        dict_ground_truth = dict()

        for index, row in self.df_ground_truth.iterrows():
            id1,id2 = row
            if id1 not in dict_ground_truth:
                dict_ground_truth[id1] = {id2}
            else:
                dict_ground_truth[id1] = dict_ground_truth[id1] | {id2}
            if id2 not in dict_ground_truth:
                dict_ground_truth[id2] = {id1}
            else:
                dict_ground_truth[id2] = dict_ground_truth[id2] | {id1}

        for node in self.graph.nodes():
            if node not in dict_ground_truth:
                dict_ground_truth[node] = {node}

        return (dict_ground_truth)

    # Given a set_query (set of entities), update dict_entity and dict_entity_belonging by querying entities in the set_query
    # then update df_benefit. type_query is "entity" or "batch"
    def query (self, set_query, type_query):
        # I HAVE FOUR KIND OF QUERIES: BATCH, entity, 
        # SKIP (THE LAST BATCH OF A COMMUNITY WHOSE LENGHT IS DIFFERENT TO batch_size), 
        # AND LAST FOR THE FINAL COMMUNITY
        #print()
        self.last_query_stats = self.empty_query_stats()

        # old_entities contains all entities that are already visited
        old_entities = set()
        # old_and_bigger_entities contains all old_entities that match with at least another element in set_query
        old_and_bigger_entities = set()
        # old_merged_entities contains all old_entities that are merged with at least another element in set_query
        old_merged_entities = set()
        # both_old_and_non_matching contains all couples of entities that are both old and they non-match
        both_old_and_non_matching = set()
        for v in set_query:
            if v not in self.dict_entity_belonging.keys():
                self.dict_entity[v] = {"set_entity" : {v}, "non_matching" : set(), "low_prob": set()}
                self.dict_entity_belonging[v] = v
            else:
                old_entities.add(v)

        if type_query in ["entity", "batch"]:
            match_pairs = self._resolve_query_matches(set_query)

            for (x,y) in itertools.combinations(set_query, 2):


                u = self.dict_entity_belonging[x]
                v = self.dict_entity_belonging[y]

                # Swich u and v if v has higher degree
                # indeed I contract v into u
                deg_u = self.multigraph.degree(u)
                deg_v = self.multigraph.degree(v)

                if deg_v > deg_u:
                    x,y = y,x
                    u = self.dict_entity_belonging[x]
                    v = self.dict_entity_belonging[y]

                if u != v:
                    # match-case

                    if self.entities_match(u, v, match_pairs):

                        # merge u and v into u, then delete v
                        self.dict_entity[u] = {"set_entity" : self.dict_entity[u]["set_entity"] | self.dict_entity[v]["set_entity"], 
                                        "non_matching" : self.dict_entity[u]["non_matching"] | self.dict_entity[v]["non_matching"],
                                        "low_prob": self.dict_entity[u]["low_prob"] | self.dict_entity[v]["low_prob"]}

                        # merge u and v into u, then delete v
                        for z in self.multigraph.neighbors(v):
                            if z != u:
                                edges = self.multigraph.get_edge_data(v,z)
                                for edge in edges.values():
                                    weight = edge["weight"]
                                    max_weight = edge["max_weight"]

                                    if self.multigraph.has_edge(u, z):
                                        edge_uz = self.multigraph.get_edge_data(u,z)
                                        edge_uz = edge_uz[0]
                                        weight_uz = edge_uz["weight"]
                                        max_weight_uz = edge_uz["max_weight"]

                                        weight_new = weight + weight_uz
                                        max_weight_new = max(max_weight, max_weight_uz)
                                        self.multigraph.remove_edge(u,z)
                                        self.multigraph.add_edge(u, z, weight = weight_new, max_weight = max_weight_new)
                                    else:
                                        self.multigraph.add_edge(u, z, weight = weight, max_weight = max_weight)

                                #self.multigraph.remove_edge(v,z)

                        self.multigraph.remove_node(v)

                        # update old_and_bigger_entities
                        if u in old_entities:
                            old_and_bigger_entities.add(u)
                        # useless because I delete v
                        if v in old_entities:
                            old_merged_entities.add(v)

                        # update the non-matching
                        for xx in self.dict_entity[v]["non_matching"]:
                            set_xx_non_matching = self.dict_entity[xx]["non_matching"]
                            if v in set_xx_non_matching:
                                set_xx_non_matching = set_xx_non_matching - self.dict_entity[u]["set_entity"]
                                set_xx_non_matching.add(u)
                                self.dict_entity[xx]["non_matching"] = set_xx_non_matching

                        # update the low_prob
                        for xx in self.dict_entity[v]["low_prob"]:
                            set_xx_low_prob = self.dict_entity[xx]["low_prob"]
                            if v in set_xx_low_prob:
                                set_xx_low_prob = set_xx_low_prob - self.dict_entity[u]["set_entity"]
                                set_xx_low_prob.add(u)
                                self.dict_entity[xx]["low_prob"] = set_xx_low_prob

                        for x in self.dict_entity[v]["set_entity"]:
                            self.dict_entity_belonging[x] = u


                        del self.dict_entity[v]

                    # NON-match-case
                    else:
                        self.dict_entity[u]["non_matching"] = self.dict_entity[u]["non_matching"] | {v}
                        self.dict_entity[v]["non_matching"] = self.dict_entity[v]["non_matching"] | {u}
                        # update both_old_and_non_matching
                        if u in old_entities and v in old_entities:
                            #both_old_and_non_matching.add({u,v})
                            both_old_and_non_matching.add(frozenset({u,v}))

        # new_entities contains the entities in set_query without duplicates
        new_entities = set()
        for v in set_query:
            new_entities.add(self.dict_entity_belonging[v])
        # remove from new_entities the old entities that are not bigger (they do not need to be updated)
        old_not_interesting = set(old_entities) - set(old_and_bigger_entities) 
        new_entities.difference_update(old_not_interesting)


        # delete the rows that contains an entity in old_and_bigger_entities or both_old_and_non_matching
        rows_to_delete = []
        old_to_delete = old_and_bigger_entities | old_merged_entities

        set_to_delete_1 = set()
        set_to_delete_2 = set()

        # AND
        for pair in both_old_and_non_matching:
            entity1, entity2 = pair
            indices = self.dict_row_df_benefit[entity1] & self.dict_row_df_benefit[entity2] # AND
            for index in indices:
                rows_to_delete.append(index)
                set_to_delete_2.add(index)

        for pair in list(itertools.combinations(old_to_delete, 2)):
            entity1, entity2 = pair
            indices = self.dict_row_df_benefit[entity1] | self.dict_row_df_benefit[entity2] # OR
            for index in indices:
                rows_to_delete.append(index)
                set_to_delete_1.add(index)


        rows_to_delete = list(set(rows_to_delete))
        for index in rows_to_delete:
            row = self.df_benefit.loc[index]
            entity1 = row["entity1"]
            entity2 = row["entity2"]
            self.dict_row_df_benefit[entity1].discard(index)
            self.dict_row_df_benefit[entity2].discard(index)

        rows_to_delete = list(set(rows_to_delete))



        self.df_benefit = self.df_benefit.drop(rows_to_delete, axis=0)
        #self.df_benefit = self.df_benefit.reset_index(drop=True)

        benefit_entities = set(self.dict_entity.keys())
        benefit_entities.difference_update(old_not_interesting) 
        if  type_query in ["last"]:
            benefit_entities.difference_update(new_entities)

        #df_benefit_concat = pd.DataFrame(columns=["entity1", "entity2", "dim1", "dim2", "comm1", "comm2", "prod_dim", "prob", "benefit", "match"])
        df_benefit_concat = pd.DataFrame(columns=["entity1", "entity2", "benefit"])
        rows = []

        self.max_index_df_benefit = self.max_index_df_benefit + 2

        max_index = [self.max_index_df_benefit]
        max_index = max_index[0]
        first_index = max_index

        # current = V(G)
        if (type_query == "last"):
 
            subgraph_new = self.graph.subgraph(set_query)


            for entity1, entity2, data in subgraph_new.edges(data=True):
                prob = data["weight"]

                rows.append({
                    "entity1": entity1,
                    "entity2": entity2, 
                    "benefit": round(prob,6)/self.max_weight,
                    }
                )
                #df_benefit_concat = pd.concat([df_benefit_concat, new_row], ignore_index=False) 

                for entity in [entity1, entity2]:
                    if entity in self.dict_row_df_benefit.keys():
                        self.dict_row_df_benefit[entity].add(max_index)
                    else:
                        self.dict_row_df_benefit[entity] = {max_index}

                self.max_index_df_benefit = self.max_index_df_benefit + 1
                max_index += 1

        if len(benefit_entities) > 0:
            for entity1 in new_entities:
                for entity2 in self.multigraph.neighbors(entity1):
                    if (entity2 in new_entities and entity1 < entity2) or (entity2 not in new_entities):

                        if entity2 in benefit_entities:

                            if entity1 not in self.dict_entity[entity2]["non_matching"] and entity1 not in self.dict_entity[entity2]["low_prob"]:
                                [benefit, prob] = self.compute_benefit (entity1, entity2)
                            
                                if prob > 0:
                                    if False and float(prob) < float(self.threshold_prob+0.001):
                                        self.dict_entity[entity1]["low_prob"] = self.dict_entity[entity1]["low_prob"] | {entity2}
                                        self.dict_entity[entity2]["low_prob"] = self.dict_entity[entity2]["low_prob"] | {entity1}
                                    else:         
                                        max_index = [self.max_index_df_benefit]
                                        rows.append({
                                                    "entity1": entity1,
                                                    "entity2": entity2, 
                                                    "benefit": round(benefit,6),
                                                    }
                                                )

                                        max_index = max_index[0]

                                        for entity in [entity1, entity2]:
                                            if entity in self.dict_row_df_benefit.keys():
                                                self.dict_row_df_benefit[entity].add(max_index)
                                            else:
                                                self.dict_row_df_benefit[entity] = {max_index}

                                        self.max_index_df_benefit = self.max_index_df_benefit + 1 

        # sort if and only if there is at least one match
        if len(rows) > 0:
            df_benefit_concat = pd.DataFrame(rows,  index=range(first_index, first_index + len(rows) ))

                
            self.df_benefit = pd.concat([self.df_benefit, df_benefit_concat], ignore_index=False) 

            self.df_benefit = self.df_benefit.sort_values(by = "benefit", ascending=False, ignore_index=False)     

        return self.last_query_stats

    # Information (number of nodes, edge, weight, degrees list, matching and percentage wrt number of edges, number of entities)
    def info_plot_community (self, comm, plot = True):

        H = self.graph.subgraph(comm)
        nodes_H = len(H.nodes())
        edges_H = len(H.edges())

        weight = self.compute_weight(set(H.nodes()))
        density = self.compute_density(set(H.nodes()))



        H_colored = nx.Graph()
        matching = 0
        matching_star = 0

        for u in H.nodes():
            for v in H.nodes():
                if u < v:
                    if v in self.dict_ground_truth[u]:
                        H_colored.add_edge(u, v, color='green')
                        matching_star += 1
                        if H.has_edge (u,v):
                            matching += 1
                    else:
                        H_colored.add_edge(u, v, color='gray')
        edge_colors = [H_colored[u][v]['color'] for u, v in H_colored.edges()]

        # GREEN GRAPH ######################

        green_edges = [(u, v) for u, v, d in H_colored.edges(data=True) if d['color'] == 'green']

        # Create a new graph that only contains the green edges
        green_graph = H_colored.edge_subgraph(green_edges).copy()

        green_components = nx.number_connected_components(green_graph)

        green_nodes = green_graph.nodes()

        other_entities = H.nodes() - green_nodes

        
        list_dimension_entity = list()
        for connected_components in nx.connected_components(green_graph):
            n = len(connected_components)
            list_dimension_entity.append(n)
        for x in other_entities:
            list_dimension_entity.append(1)
        
        return(nodes_H, matching, matching_star)
        
    # Weight of a community
    def compute_weight_community (self, community):
        return (self.compute_weight(community))

    # create self.list_community (it also calls self.create_dict_comm)
    def create_list_community (self):

        weight_threshold = self.lambda_w

        if weight_threshold == "False":
            self.list_community = [set(self.graph.nodes())]

            # create the dictionaries self.dict_community and self.dict_community_inverted
            self.create_dict_comm()
            return ()


        if self.alg_community in ["leiden", "infomap"]:
            self.convert_to_igraph()



        # add also element of df_ground_truth if there are isolated vertices
        list_community_light = [self.graph.nodes()] 
        list_community_heavy = list()
        list_community_small = list()
        #list_weight_community_heavy = list()

        if self.alg_community == "DBSCAN":
            self.create_dist_matrix()


        while len(list_community_light) > 0:

            # some prints step by step
            if False:
                list_len_light_comm = list()
                for comm in list_community_light:
                    list_len_light_comm.append(len(comm))
                print("light: ", list_len_light_comm)

               
            # divide in light, heavy and small
            list_community_light_NEW = list()
            nodes_graph = self.graph.nodes()
            for comm_light in list_community_light:
                comm_light = list(comm_light)
                
                if self.alg_community in ["leiden", "infomap"]:
                    comm_light_subgraph = self.igraph.subgraph(comm_light)
                else:
                    comm_light_subgraph = self.graph.subgraph(comm_light)


                if self.alg_community == "DBSCAN":
                    min_samples = 2
                    eps_dbscan = 5

                    db = DBSCAN(eps = eps_dbscan, min_samples = min_samples, metric='precomputed')  # adjust eps based on your scale
                    idx = list(comm_light)

                    submatrix = self.dist_matrix[np.ix_(idx, idx)]
                    labels = db.fit_predict(submatrix)

                    comms_dict = {}
                    for node, label in zip(nodes_graph, labels):
                        comms_dict.setdefault(label, []).append(node)

                    comms = [comms_dict[k] for k in sorted(comms_dict.keys())]
                    
                    while (len(comms) < 2):
                        eps_dbscan -= 0.1
                        db = DBSCAN(eps = eps_dbscan, min_samples = min_samples, metric='precomputed')
                        labels = db.fit_predict(submatrix)

                        comms_dict = {}
                        for node, label in zip(nodes_graph, labels):
                            comms_dict.setdefault(label, []).append(node)

                        comms = [comms_dict[k] for k in sorted(comms_dict.keys())]



                if self.alg_community == "louvain":
                    
                    max_level = None
                    if len(self.graph.nodes()) > 20000:
                        resolution = 0.50
                    else:
                        resolution = 0.75

                    comms = nx.community.louvain_communities(comm_light_subgraph, weight="weight", resolution=resolution, threshold=1e-04, max_level = max_level)
                    

                if self.alg_community == "leiden":
                
                    resolution = 5
                    partition = leidenalg.find_partition(comm_light_subgraph, leidenalg.ModularityVertexPartition, weights='weight')
                    comms =  [set(comm_light[idx] for idx in community) for community in partition]


                if self.alg_community == "infomap":

                    trials = 10
                    partition = comm_light_subgraph.community_infomap(edge_weights = "weight", trials = trials)
                    comms =  [set(comm_light[idx] for idx in community) for community in partition]


                if self.alg_community == "modularity":
                    resolution = 0.30
                    comms = nx.community.greedy_modularity_communities(comm_light_subgraph, weight="weight", resolution=resolution)
                    while (len(comms) < 2):
                        resolution += 0.2
                        comms = nx.community.greedy_modularity_communities(comm_light_subgraph, weight="weight", resolution=resolution)


                if self.alg_community == "lpa":
                    seed = 0
                    comms = nx.community.asyn_lpa_communities(comm_light_subgraph, weight="weight", seed=seed)
                    comms = list(comms)
                    while (len(comms) < 2) and seed < 10:
                        seed += 1
                        comms = nx.community.asyn_lpa_communities(comm_light_subgraph, weight="weight", seed=seed)
                        comms = list(comms)

                list_comm_big = list()
                list_comm_small = list()


                # ATTENTION! NOW list_community_small CONTAINS ALSO INDIVISIBLE COMMS
                # SEE OR  len(comms) == 1
                for comm in comms:
                    if len(comm) < max(self.batch_size, 10) or len(comms) == 1:
                        list_community_small.append(list(comm))
                    else:
                        density = self.compute_density(comm)
                        if density >= weight_threshold:
                            list_community_heavy.append(list(comm))
                        else:
                            list_community_light_NEW.append(list(comm))

            list_community_light = list_community_light_NEW


        # Consider only the heavy communities
        self.sum_heavy_comm = 0
        for comm in list_community_heavy:
            self.sum_heavy_comm += len(comm)
        self.number_heavy = len(list_community_heavy)

        last_comm = set()

        if (len(list_community_small) > 0):
            for comm in list_community_small:
                last_comm.update(set(comm))

        partial_fun = functools.partial(self.compute_weight_community)

        # sort the communities
        self.list_community = sorted(list_community_heavy, key=partial_fun, reverse=True)

        if len(last_comm) > 0:
            self.list_community.append(last_comm)

        # create the dictionaries self.dict_community and self.dict_community_inverted
        self.create_dict_comm()


    # create self.dict_community and self.dict_community_inverted
    def create_dict_comm (self):
        i = 0
        self.dict_community = dict()
        self.dict_community_inverted = dict()
        for c in self.list_community:
            self.dict_community[i] = set()
            for x in c:
                self.dict_community[i].add(x)
                self.dict_community_inverted[x] = i
            i += 1

    # compute and print the recall and the recall_community
    def compute_recall_start(self):
        recall_count = 0
        for i,row in self.df_ground_truth.iterrows():
            entity1,entity2 = row
            if self.graph.has_edge(entity1,entity2):
                recall_count += 1

        self.recall = recall_count/len(self.df_ground_truth)


        recall_community_count = 0
        recall_community_max_count = 0

        if self.alg_community == "no":
            self.recall_community_max = 1
            self.recall_community = self.recall
        else:
            for i,row in self.df_ground_truth.iterrows():
                [x,y] = row
                if self.dict_community_inverted[x] == self.dict_community_inverted[y]:
                    recall_community_max_count += 1
                    if self.graph.subgraph(self.dict_community[self.dict_community_inverted[x]]).has_edge(x,y):
                        recall_community_count += 1

        self.recall_community_max = recall_community_max_count/len(self.df_ground_truth)
        self.recall_community = recall_community_count/len(self.df_ground_truth)


        print("recall = ", round(self.recall,3))
        print("recall community = ", round(self.recall_community,3))
        print("recall max community = ", round(self.recall_community_max,3))

        print("number of communities: ", len(self.list_community))

        len_comm = list()
        for comm in self.list_community:
            len_comm.append(len(comm))
        len_comm.sort(reverse=True)
        print("dimension of communities:", len_comm)

        weight_comm = list()
        for comm in self.list_community:
            sub_comm = self.graph.subgraph(comm)
            nodes_sub = set(sub_comm.nodes())
            if len(nodes_sub) > 1:
                weight = self.compute_weight_community(nodes_sub)
            else:
                weight = -1
            weight_comm.append(round(weight,6))
        print("weight of communities:", weight_comm)

    # return the entities whose delta_benefit is higher than the temperature
    def compute_entity_higher_temperature(self):

        if len(self.df_benefit) == 0:
            return (list())

        if self.batch_size == 2:
            u,v,benefit = self.df_benefit.loc[self.df_benefit.index[0]]
            if benefit > self.temperature:
                return([u,v])
            else:
                return(list())




        df_aux = self.df_benefit.head(1000)
        df_aux = df_aux[df_aux["benefit"] >= self.temperature]




        set_all_entity = set(df_aux["entity1"]) | set(df_aux["entity2"])
        # if there are few entities, then return
        if len(set_all_entity) <= self.batch_size:
            return(set_all_entity)

        graph_aux = nx.Graph()
        for i, row in df_aux.iterrows():
            entity1, entity2, weight = row[["entity1", "entity2", "benefit"]]
            graph_aux.add_edge(entity1, entity2, weight=weight)

        vertex_weight_sum = {node: sum(data['weight'] for _, _, data in graph_aux.edges(node, data=True)) for node in graph_aux.nodes()}

        # Sort all edges by weight
        sorted_edges = sorted(graph_aux.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)

        dict_selected_vertices = {node: 0 for node in graph_aux.nodes()}

        return(self.greedy_heaviest_subgraph(graph_aux, vertex_weight_sum, sorted_edges, dict_selected_vertices))

    # return the entities whose delta_benefit is higher than the temperature
    def greedy_heaviest_subgraph(self, graph_aux, vertex_weight_sum, sorted_edges, dict_selected_vertices):

        max_vertex = max(vertex_weight_sum, key=vertex_weight_sum.get)
        
        set_vertices = {max_vertex}

        set_neigh = set(graph_aux.neighbors(max_vertex))

        # Find second vertex, and thus first edge
        max_weight = -1
        max_vertex_2 = -1
        for x in graph_aux.neighbors(max_vertex):
            weight_x = graph_aux.get_edge_data(max_vertex,x)["weight"]
            if weight_x > max_weight:
                max_vertex_2 = x
                max_weight = weight_x

        # if max_vertex_2 == -1, then it means that there are not edges!!!
        if max_vertex_2 == -1:
            return(list(graph_aux.nodes()))



        set_vertices.add(max_vertex_2)
        set_neigh.update(set(graph_aux.neighbors(max_vertex_2)))
        set_neigh.remove(max_vertex_2)

        graph_aux.remove_edge(max_vertex, max_vertex_2)

        


        while len(set_vertices) < self.batch_size and len(graph_aux.edges()) > 0:
            time_old = time.time()
            max_weight = -1
            # Compute the max vertex, by default I choose the first vertex in the best edge
            max_vertex_x = list(set(graph_aux.nodes()) - set_vertices)[0]
            
            for x in set_neigh:
                weight_x = 0
                for y in set_vertices:
                    weight_xy = graph_aux.get_edge_data(x,y)
                    if weight_xy:
                        weight_x += weight_xy["weight"]
                if weight_x > max_weight:
                    max_weight = weight_x
                    max_vertex_x = x
            
            acceptable = 0
            index_acceptable = 0
            while acceptable == 0 and index_acceptable != len(sorted_edges) - 1:
                max_edge_x, max_edge_y, max_edge_weight = sorted_edges[index_acceptable]
                if index_acceptable == len(sorted_edges) - 1 or ((dict_selected_vertices[max_edge_x] == 0 or dict_selected_vertices[max_edge_y] == 0) and graph_aux.has_edge(max_edge_x, max_edge_y) and (max_edge_x not in set_vertices and max_edge_y not in set_vertices)):
                    acceptable = 1
                else:
                    index_acceptable += 1
            if index_acceptable == len(sorted_edges) - 1:
                edge_are_finished = 1
            else:
                edge_are_finished = 0

            max_edge_weight = max_edge_weight["weight"]
            
            # If the max edge is better, then add it
            if max_edge_weight > max_weight and len(set_vertices) <= self.batch_size-2 and edge_are_finished == 0:
                if max_edge_x in set_vertices or max_edge_y in set_vertices: # Check if both vertices of max_edge are new
                    ERRORE 
                set_vertices.update({max_edge_x, max_edge_y})
                set_neigh.discard({max_edge_x, max_edge_y})
                set_neigh.update(set(graph_aux.neighbors(max_edge_x)))
                set_neigh.update(set(graph_aux.neighbors(max_edge_y)))
                set_neigh = set_neigh - set_vertices

                vertex_common_x = set(graph_aux.neighbors(max_edge_x)) & set_vertices
                for z in vertex_common_x:
                    graph_aux.remove_edge(max_edge_x, z)
                vertex_common_y = set(graph_aux.neighbors(max_edge_y)) & set_vertices
                for z in vertex_common_y:
                    graph_aux.remove_edge(max_edge_y, z)

            else:
                vertex_common = set(graph_aux.neighbors(max_vertex_x)) & set_vertices
                set_vertices.add(max_vertex_x)
                set_neigh.discard(max_vertex_x)
                set_neigh.update(set(graph_aux.neighbors(max_vertex_x)))
                set_neigh = set_neigh - set_vertices
                for z in vertex_common:
                    graph_aux.remove_edge(max_vertex_x, z)
          
        return(list(set_vertices))

    # compute the recall and the number of found matchings
    def compute_recall(self):
        metrics = self.compute_metrics()
        return(metrics["recall"], metrics["true_positive_pairs"])

    # check if the nodes in the new batch are in the same community of the previous batch
    # Attention! Use only for batch queries
    # DEPRECATED
    def change_community (self, batch, old_set_community):
        new_set_community = set()
        for x in batch:
            new_set_community.add(self.dict_community_inverted[x])

        if new_set_community <= old_set_community:
            return(new_set_community, False)
        else:
            return(new_set_community, True)

    def create_dist_matrix (self):

        def graph_to_inverse_weight_matrix(G, default_distance=100.0):
            nodes = list(G.nodes())
            n = len(nodes)
            dist_matrix = np.full((n, n), default_distance)
            for i, j, data in G.edges(data=True):
                weight = data.get('weight', 1.0)
                distance = 1.0 / (weight + 0.00001)

                dist_matrix[i, j] = distance
                dist_matrix[j, i] = distance  # for undirected graphs

            np.fill_diagonal(dist_matrix, 0.0)

            return dist_matrix  # return nodes in case you want labels

        G = self.graph
        dist_matrix = graph_to_inverse_weight_matrix(G)

        shortest_paths = shortest_path(dist_matrix, directed=False, unweighted=False)

        n = len(G.nodes())
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i,n):
                if not G.has_edge(i, j):
                    G.add_edge(i, j)
                    G[i][j]['weight'] = 0
                
                short = shortest_paths[i][j]
                distance_matrix[i][j] = short
                distance_matrix[j][i] = short

        self.dist_matrix = distance_matrix

    def convert_to_igraph (self):
        # Convert to igraph
        G_nx = self.graph
        mapping = {node: idx for idx, node in enumerate(G_nx.nodes())}

        edges = [(mapping[u], mapping[v]) for u, v in G_nx.edges()]

        weights = [G_nx[u][v].get('weight') for u, v in G_nx.edges()]

        G_ig = ig.Graph(edges=edges, directed=False)
        G_ig.add_vertices(len(G_nx.nodes()) +1 - G_ig.vcount())
        G_ig.es['weight'] = weights

        self.igraph = G_ig


