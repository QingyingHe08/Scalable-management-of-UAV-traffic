
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 20:28:46 2025

@author: qingy
"""

import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import math
from collections import defaultdict
from pyproj import Transformer



def make_symmetric(dist_dict):
    """
    if (i, j) exists but (j, i) does not exist, then let (j, i) = (i, j).
    """
    symmetric_dict = dist_dict.copy()
    for (i, j), d in list(dist_dict.items()):
        if (j, i) not in symmetric_dict:
            symmetric_dict[(j, i)] = d
    return symmetric_dict


def compute_node_distances_subset(df_nodes, pair_keys, symmetric=False, skip_missing=True):
    """
    return
    ----
    dict: {(u,v): distance, ...}
    """

    coord_map = {
        nid: np.array([x, y], dtype=float)
        for nid, x, y in df_nodes[['node_id', 'x_km', 'y_km']].itertuples(index=False, name=None)
    }

    distances = {}

    computed = set()

    for u, v in pair_keys:
        key_undirected = (u, v) if u <= v else (v, u)
        if key_undirected in computed:

            if symmetric and (v, u) not in distances and (u, v) in distances:
                distances[(v, u)] = distances[(u, v)]
            continue

        if u not in coord_map or v not in coord_map:
            if skip_missing:
                continue
            else:
                raise KeyError(f"Node id not found in df_nodes: {u if u not in coord_map else v}")

        dist = np.linalg.norm(coord_map[u] - coord_map[v])
        distances[(u, v)] = dist
        if symmetric:
            distances[(v, u)] = dist

        computed.add(key_undirected)

    return distances




def prepare_data(df_orders, top_node_ids, df_nodes, dist_dict, radius_km=3.0):


    df_nodes_top = df_nodes[df_nodes['node_id'].isin(top_node_ids)].reset_index(drop=True)
    df_nodes_top_ids = set(df_nodes_top['node_id'].tolist())

    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(df_nodes_top['node_id'])}
    idx_to_node_id = {idx: node_id for node_id, idx in node_id_to_idx.items()}

    n = len(df_nodes_top)

    adj_matrix = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(adj_matrix, 0.0)


    links = []
    for (i, j), dist in dist_dict.items():
        if i in df_nodes_top_ids and j in df_nodes_top_ids:
            links.append({'from_node': i, 'to_node': j, 'distance_km': dist})
            adj_matrix[node_id_to_idx[i], node_id_to_idx[j]] = dist

    df_links = pd.DataFrame(links)


    cols_needed = ['order_id', 'pickup_id', 'dropoff_id', 'fetch_time', 'arrive_time', 'distance_km']
    missing_cols = [c for c in cols_needed if c not in df_orders.columns]
    if missing_cols:
        raise KeyError(f"df_orders 缺少所需列: {missing_cols}")

    df_trips = df_orders[
        (df_orders['pickup_id'].isin(df_nodes_top_ids)) &
        (df_orders['dropoff_id'].isin(df_nodes_top_ids))
    ][cols_needed].copy()


    order_pairs = set(zip(df_trips['pickup_id'], df_trips['dropoff_id']))
    link_pairs = set((row['from_node'], row['to_node']) for _, row in df_links.iterrows())
    missing_for_orders = order_pairs - link_pairs
    if missing_for_orders:

        print(f"[Info] 共有 {len(missing_for_orders)} 个订单的 (pickup, dropoff) 不在 dist_dict/links 中，例如: {list(missing_for_orders)[:5]}")


    df_trips['bid'] = df_trips['distance_km'] * 1.5

    return df_nodes_top, df_links, df_trips, node_id_to_idx, idx_to_node_id, adj_matrix



import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

def find_shortest_paths(df_links, node_id_to_idx, idx_to_node_id, dist_dict):
    """
    return:
        shortest_paths: dict[(from_id, to_id)] = {
            'path': [node_ids...],
            'distance': float,
            'links': [(u,v), ...]
        }
    """
    shortest_paths = {}

    for (u, v), dist in dist_dict.items():
        if u == v:
            shortest_paths[(u, v)] = {
                'path': [u],
                'distance': 0.0,
                'links': []
            }
        else:
            shortest_paths[(u, v)] = {
                'path': [u, v],
                'distance': float(dist),
                'links': [(u, v)]
            }

    return shortest_paths




# 3.
def solve_offline_optimization_with_gurobi(df_trips, df_links, shortest_paths,
                                          uav_speed=45.0, time_step_min=1, capacity=3):


    #
    required_columns = ['order_id', 'pickup_id', 'dropoff_id', 'fetch_time', 'bid']
    for col in required_columns:
        if col not in df_trips.columns:
            raise ValueError(f"Missing required column: {col}")

    #
    min_fetch_time = df_trips['fetch_time'].min()
    max_arrive_time = df_trips['arrive_time'].max()
    total_time_steps = int((max_arrive_time - min_fetch_time) / 60 / time_step_min) + 100

    #
    model = gp.Model("UAV_Offline_Optimization")
    model.setParam("OutputFlag", 1)
    model.setParam("TimeLimit", 600)

    #
    x = model.addVars(len(df_trips), vtype=GRB.BINARY, name="x")
    y_vars = {}

    #
    model.setObjective(gp.quicksum(x[i] * df_trips.iloc[i]['bid'] for i in range(len(df_trips))), GRB.MAXIMIZE)

    #
    order_id_to_idx = {oid: idx for idx, oid in enumerate(df_trips['order_id'])}

    j = 0
    for idx, order in df_trips.iterrows():

        i = idx

        pickup = order['pickup_id']
        dropoff = order['dropoff_id']

        if (pickup, dropoff) not in shortest_paths:
            model.addConstr(x[j] == 0, name=f"no_path_{i}")
            continue

        path_info = shortest_paths[(pickup, dropoff)]
        path_links = path_info['links']
        travel_time_min = path_info['distance'] / (uav_speed / 60.0)
        travel_time_steps = int(np.ceil(travel_time_min / time_step_min))


        start_time = int((order['fetch_time'] - min_fetch_time) / 60 / time_step_min)


        for link in path_links:

            if link not in df_links[['from_node', 'to_node']].apply(tuple, axis=1).tolist():
                link = (link[1], link[0])

            for t in range(start_time, start_time + travel_time_steps):
                if t < total_time_steps:

                    if (link, t) not in y_vars:
                        y_vars[(link, t)] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"y_{link[0]}_{link[1]}_{t}")


                    model.addConstr(y_vars[(link, t)] >= x[j], name=f"x_link_time_{i}_{link}_{t}")

        j = j + 1


    for (link, t), y in y_vars.items():
        model.addConstr(y <= capacity, name=f"capacity_{link}_{t}")


    model.optimize()


    accepted_orders = []
    for i in range(len(df_trips)):
        if x[i].X > 0.5:

            accepted_orders.append(df_trips.iloc[i]['order_id'])

    total_welfare = model.ObjVal
    acceptance_rate = len(accepted_orders) / len(df_trips) if len(df_trips) > 0 else 0

    return {
        'status': model.Status,
        'objective': total_welfare,
        'accepted_orders': accepted_orders,
        'acceptance_rate': acceptance_rate,
        'solve_time': model.Runtime
    }


def read_meituan_orders(file_path):
    """
    read dataset
    """
    df = pd.read_csv(file_path)

    required_fields = [
        'dispatch_time', 'grab_time', 'estimate_arrived_time', 'estimate_meal_prepare_time',
        'fetch_time', 'arrive_time',
        'sender_lng', 'sender_lat',
        'recipient_lng', 'recipient_lat',
        'grab_lng', 'grab_lat'
    ]
    df = df.dropna(subset=required_fields)
    df = df[df['fetch_time'] > 0]
    df = df[df['arrive_time'] > df['fetch_time']]
    df = df.reset_index(drop=True)

    return df

def convert_coords_to_km(df_nodes):

    df_nodes["lon"] = df_nodes["x"] / 1e6
    df_nodes["lat"] = df_nodes["y"] / 1e6

    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    x_m, y_m = transformer.transform(df_nodes["lon"].values, df_nodes["lat"].values)

    df_nodes["x_km"] = x_m / 1000
    df_nodes["y_km"] = y_m / 1000

    return df_nodes[["node_id", "x_km", "y_km"]]

def build_node_and_trip_tables(df_orders):


    pickup_nodes = df_orders[["sender_lng", "sender_lat"]].rename(columns={"sender_lng": "x", "sender_lat": "y"})
    drop_nodes = df_orders[["recipient_lng", "recipient_lat"]].rename(columns={"recipient_lng": "x", "recipient_lat": "y"})
    all_nodes = pd.concat([pickup_nodes, drop_nodes]).drop_duplicates().reset_index(drop=True)
    all_nodes["node_id"] = all_nodes.index
    df_nodes_raw = all_nodes[["node_id", "x", "y"]]


    df_nodes_km = convert_coords_to_km(df_nodes_raw)


    coord2id = {(row.x, row.y): row.node_id for row in df_nodes_raw.itertuples(index=False)}
    id2coord_km = {row.node_id: (row.x_km, row.y_km) for row in df_nodes_km.itertuples(index=False)}


    df_trips = pd.DataFrame({
        "pickup_id": df_orders.apply(lambda r: coord2id[(r.sender_lng, r.sender_lat)], axis=1),
        "dropoff_id": df_orders.apply(lambda r: coord2id[(r.recipient_lng, r.recipient_lat)], axis=1),
        "fetch_time": df_orders["fetch_time"],
        "arrive_time": df_orders["arrive_time"],
        "order_id": df_orders.index
    })


    def euclidean_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    df_trips["distance_km"] = df_trips.apply(
        lambda r: euclidean_distance(id2coord_km[r.pickup_id], id2coord_km[r.dropoff_id]), axis=1
    )
    df_trips["duration_sec"] = df_trips["arrive_time"] - df_trips["fetch_time"]
    df_trips = df_trips.reset_index().rename(columns={"index": "order_id"})  # 添加订单ID

    return df_nodes_km, df_trips


def check_node_occurrences(df_trips):

    pickup_counts = df_trips["pickup_id"].value_counts().rename("pickup_count")
    dropoff_counts = df_trips["dropoff_id"].value_counts().rename("dropoff_count")


    node_counts = pd.concat([pickup_counts, dropoff_counts], axis=1).fillna(0).astype(int)
    node_counts["total_count"] = node_counts["pickup_count"] + node_counts["dropoff_count"]
    node_counts = node_counts.reset_index().rename(columns={"index": "node_id"})


    duplicated_nodes = node_counts[node_counts["total_count"] > 1]

    return node_counts, duplicated_nodes

class OnlineAllocator:
    def __init__(self, df_orders, df_links, shortest_paths, uav_speed=45, time_step_min=1, capacity=1):
        """
        Initialization
        """

        min_fetch_time = df_orders['fetch_time'].min()
        self.min_fetch_time = min_fetch_time
        self.time_step_min = time_step_min


        self.df_orders = df_orders.sort_values('fetch_time').reset_index(drop=True)
        self.df_orders['start_time_step'] = self.df_orders['fetch_time'].apply(
               lambda t: int((t - min_fetch_time) / 60 / time_step_min)
           )

        # basi paramters
        self.uav_speed = uav_speed
        self.capacity = capacity
        self.uav_speed_km_per_min = uav_speed / 60.0  # km/min


        self.shortest_paths = shortest_paths
        self.df_links = df_links


        self.link_capacity = defaultdict(lambda: defaultdict(lambda: capacity))


        self.results = {
            'accepted': [],
            'rejected': [],
            'total_welfare': 0,
            'acceptance_rate': 0,
            'runtime': 0
        }

    def reset(self):

        self.link_capacity = defaultdict(lambda: defaultdict(lambda: self.capacity))
        self.results = {
            'accepted': [],
            'rejected': [],
            'total_welfare': 0,
            'acceptance_rate': 0,
            'runtime': 0
        }

    def get_link_distance(self, link):

        from_node, to_node = link

        # 尝试从 df_links 中找到距离
        try:
            row = self.df_links[((self.df_links['from_node'] == from_node) &
                                 (self.df_links['to_node'] == to_node)) |
                                ((self.df_links['from_node'] == to_node) &
                                 (self.df_links['to_node'] == from_node))].iloc[0]
            return row['distance_km']
        except IndexError:

            return 0.5

    def check_path_availability(self, order, start_time_step):

        pickup = order['pickup_id']
        dropoff = order['dropoff_id']


        if (pickup, dropoff) not in self.shortest_paths:
            return False, 0, []

        path_info = self.shortest_paths[(pickup, dropoff)]
        path_links = path_info['links']


        travel_time_min = path_info['distance'] / self.uav_speed_km_per_min
        travel_time_steps = math.ceil(travel_time_min / self.time_step_min)


        link_time_usage = []
        current_time = start_time_step


        for link in path_links:

            dist = self.get_link_distance(link)
            link_travel_time_min = dist / self.uav_speed_km_per_min
            link_travel_steps = max(1, math.ceil(link_travel_time_min / self.time_step_min))


            for step in range(int(current_time), int(current_time + link_travel_steps)):
                if self.link_capacity[link].get(step, self.capacity) <= 0:
                    return False, 0, []
                link_time_usage.append((link, step))

            current_time += link_travel_steps

        return True, travel_time_steps, link_time_usage

    def allocate_order(self, order_idx, order):

        raise NotImplementedError("Subclasses must implement allocate_order method")

    def run(self):

        start_time = time.time()


        for order_idx, order in self.df_orders.iterrows():
            self.allocate_order(order_idx, order)


        total_orders = len(self.df_orders)
        accepted_count = len(self.results['accepted'])
        self.results['acceptance_rate'] = accepted_count / total_orders if total_orders > 0 else 0
        self.results['runtime'] = time.time() - start_time

        return self.results



import time
import copy
import numpy as np
from collections import defaultdict


def scale_capacity(alloc, factor=0.6):
    for link, ts_dict in alloc.link_capacity.items():
        for ts in list(ts_dict.keys()):
            old = ts_dict[ts]
            ts_dict[ts] = max(1, int(old * factor))


class FCFSAllocator(OnlineAllocator):

    def allocate_order(self, order_idx, order):

        start_time_step = order['start_time_step']

        available, travel_time_steps, link_time_usage = self.check_path_availability(order, start_time_step)

        if available:

            for link, time_step in link_time_usage:
                self.link_capacity[link][time_step] -= 1


            self.results['accepted'].append(order_idx)
            self.results['total_welfare'] += order['bid']
        else:
            self.results['rejected'].append(order_idx)


class GreedyAllocator(OnlineAllocator):

    def __init__(self, *args, time_window_min=5, **kwargs):

        super().__init__(*args, **kwargs)
        self.time_window_min = time_window_min
        self.time_window_steps = int(time_window_min / self.time_step_min)
        self.pending_orders = []

    def run(self):

        start_time = time.time()
        current_time_step = 0


        for order_idx, order in self.df_orders.iterrows():

            order_time_step = order['start_time_step']


            if order_time_step >= current_time_step + self.time_window_steps:
                self.process_pending_orders(current_time_step)
                current_time_step += self.time_window_steps


            self.pending_orders.append((order_idx, order))


        self.process_pending_orders(current_time_step)


        total_orders = len(self.df_orders)
        accepted_count = len(self.results['accepted'])
        self.results['acceptance_rate'] = accepted_count / total_orders if total_orders > 0 else 0
        self.results['runtime'] = time.time() - start_time

        return self.results

    def process_pending_orders(self, start_time_step):

        if not self.pending_orders:
            return


        self.pending_orders.sort(key=lambda x: x[1]['bid'], reverse=True)


        for order_idx, order in self.pending_orders:

            available, travel_time_steps, link_time_usage = self.check_path_availability(
                order, start_time_step
            )

            if available:

                for link, time_step in link_time_usage:
                    self.link_capacity[link][time_step] -= 1


                self.results['accepted'].append(order_idx)
                self.results['total_welfare'] += order['bid']
            else:
                self.results['rejected'].append(order_idx)


        self.pending_orders = []


class PrimalDualAllocator(GreedyAllocator):

    def __init__(self, *args, initial_price=1e-4, gamma=3.0, time_window_min=10, **kwargs):

        super().__init__(*args, time_window_min=time_window_min, **kwargs)


        self.base_price = float(initial_price)
        self.gamma = float(gamma)


        self.link_capacity_max = defaultdict(dict)
        for link, ts_dict in self.link_capacity.items():
            for ts, cap in ts_dict.items():
                self.link_capacity_max[link][ts] = cap

        self._pd_debug_once = False

    def _ensure_cap_max(self, link, ts):

        if ts not in self.link_capacity_max[link]:
            cap_now = self.link_capacity[link].get(ts, 0)
            self.link_capacity_max[link][ts] = max(1, cap_now)

    def resource_price(self, link, ts):

        self._ensure_cap_max(link, ts)
        cap_max = self.link_capacity_max[link][ts]
        cap_now = self.link_capacity[link].get(ts, 0)
        used = max(0, cap_max - cap_now)
        util = used / cap_max if cap_max > 0 else 1.0
        return self.base_price * (np.exp(self.gamma * util) - 1.0)

    def auto_calibrate_base_price(self, sample_orders=200, gamma=None, density_baseline=0.7):

        if gamma is not None:
            self.gamma = float(gamma)

        n = min(sample_orders, len(self.df_orders))
        if n == 0:
            return
        sample = self.df_orders.sort_values('start_time_step').reset_index(drop=True).iloc[:n]

        bids, lens = [], []
        window_start_time_step = 0
        for _, order in sample.iterrows():
            earliest = max(int(order['start_time_step']), int(window_start_time_step))
            #earliest = int(window_start_time_step)
            available, _, link_time_usage = self.check_path_availability(order, earliest)
            if available and link_time_usage:
                bids.append(float(order['bid']))
                lens.append(len(link_time_usage))

        if not bids or not lens:
            return

        B_typ = float(np.median(bids))
        L_typ = float(np.median(lens))
        if L_typ <= 0:
            return

        target_per_resource = (B_typ / L_typ) * float(density_baseline)
        util_target = 0.8
        denom = np.exp(self.gamma * util_target) - 1.0
        if denom <= 0:
            return

        self.base_price = target_per_resource / denom
        print(f'[PD-CALIBRATE] base_price={self.base_price:.3e}, gamma={self.gamma}, '
              f'B_typ={B_typ:.3f}, L_typ={L_typ:.1f}, target_per_resource={target_per_resource:.3f}')

    def process_pending_orders(self, start_time_step):

        if not self.pending_orders:
            return


        self.pending_orders.sort(key=lambda x: x[1]['bid'], reverse=True)

        for order_idx, order in self.pending_orders:
            earliest = max(int(order['start_time_step']), int(start_time_step))
            #earliest = int(start_time_step)
            available, travel_time_steps, link_time_usage = self.check_path_availability(order, earliest)
            if not available or not link_time_usage:
                self.results['rejected'].append(order_idx)
                continue


            if not self._pd_debug_once:
                path_len = len(link_time_usage)
                first_prices = [self.resource_price(*lt) for lt in link_time_usage[:3]]
                total_cost_full = sum(self.resource_price(*lt) for lt in link_time_usage)
                ratio = total_cost_full / max(float(order['bid']), 1e-9)
                print('[PD-DEBUG] bid=', float(order['bid']),
                      ' path_len=', path_len,
                      ' first_3_prices=', [round(p, 6) for p in first_prices],
                      ' total_cost≈', round(total_cost_full, 6),
                      ' ratio(cost/bid)=', round(ratio, 3))
                self._pd_debug_once = True


            total_cost = sum(self.resource_price(*lt) for lt in link_time_usage)
            if float(order['bid']) >= total_cost:
                for link, ts in link_time_usage:
                    self.link_capacity[link][ts] -= 1
                self.results['accepted'].append(order_idx)
                self.results['total_welfare'] += float(order['bid'])
            else:
                self.results['rejected'].append(order_idx)

        self.pending_orders = []



def compare_online_algorithms(df_trips_prepared, df_links, shortest_paths):

    results = {}

    # FCFS
    print("Running FCFS algorithm...")
    fcfs_allocator = FCFSAllocator(
        df_trips_prepared,
        df_links,
        shortest_paths,
        uav_speed=45,
        time_step_min=1,
        capacity=1
    )
    fcfs_results = fcfs_allocator.run()
    results['FCFS'] = fcfs_results

    # Greedy
    print("Running Greedy algorithm...")
    greedy_allocator = GreedyAllocator(
        df_trips_prepared,
        df_links,
        shortest_paths,
        uav_speed=45,
        time_step_min=1,
        capacity=1,
        time_window_min=10
    )
    greedy_results = greedy_allocator.run()
    results['Greedy'] = greedy_results


    print("Running Primal-Dual algorithm...")

    pd_allocator = PrimalDualAllocator(
        df_trips_prepared,
        df_links,
        shortest_paths,
        uav_speed=45,
        time_step_min=1,
        capacity=1,
        time_window_min=10,  # 5分钟时间窗口
        initial_price=1e-5

    )
    pd_results = pd_allocator.run()
    results['Primal-Dual'] = pd_results

    return results


def plot_comparison_results(results):

    algorithms = list(results.keys())
    welfares = [results[algo]['total_welfare'] for algo in algorithms]
    acceptance_rates = [results[algo]['acceptance_rate'] * 100 for algo in algorithms]
    runtimes = [results[algo]['runtime'] for algo in algorithms]

    plt.figure(figsize=(15, 5))


    plt.subplot(131)
    plt.bar(algorithms, welfares, color=['blue', 'green', 'orange'])
    plt.title('Total Welfare Comparison')
    plt.ylabel('Total Welfare')
    plt.grid(axis='y')


    plt.subplot(132)
    plt.bar(algorithms, acceptance_rates, color=['blue', 'green', 'orange'])
    plt.title('Acceptance Rate Comparison')
    plt.ylabel('Acceptance Rate (%)')
    plt.grid(axis='y')


    plt.subplot(133)
    plt.bar(algorithms, runtimes, color=['blue', 'green', 'orange'])
    plt.title('Runtime Comparison')
    plt.ylabel('Runtime (seconds)')
    plt.grid(axis='y')

    plt.tight_layout()
    plt.savefig('online_algorithm_comparison.png')
    plt.show()


    results_df = pd.DataFrame({
        'Algorithm': algorithms,
        'Total Welfare': welfares,
        'Acceptance Rate (%)': acceptance_rates,
        'Runtime (s)': runtimes
    })

    print("\nAlgorithm Performance Comparison:")
    print(results_df)



def visualize_results(df_trips, df_links, accepted_orders, node_id_to_idx, df_nodes_top):

    df_accepted = df_trips[df_trips['order_id'].isin(accepted_orders)]
    df_rejected = df_trips[~df_trips['order_id'].isin(accepted_orders)]


    plt.figure(figsize=(12, 10))

    plt.subplot(221)
    plt.hist(df_trips['bid'], bins=20, alpha=0.7, label='All Orders')
    plt.hist(df_accepted['bid'], bins=20, alpha=0.7, label='Accepted')
    plt.xlabel('Bid Amount')
    plt.ylabel('Count')
    plt.title('Bid Distribution')
    plt.legend()
    plt.grid(True)


    plt.subplot(222)
    plt.hist(df_trips['distance_km'], bins=20, alpha=0.7, label='All Orders')
    plt.hist(df_accepted['distance_km'], bins=20, alpha=0.7, label='Accepted')
    plt.xlabel('Distance (km)')
    plt.ylabel('Count')
    plt.title('Distance Distribution')
    plt.legend()
    plt.grid(True)


    plt.subplot(223)

    plt.scatter(df_nodes_top['x_km'], df_nodes_top['y_km'], s=50, c='blue', alpha=0.6)
    for _, link in df_links.iterrows():
        node1 = df_nodes_top[df_nodes_top['node_id'] == link['from_node']].iloc[0]
        node2 = df_nodes_top[df_nodes_top['node_id'] == link['to_node']].iloc[0]
        plt.plot([node1['x_km'], node2['x_km']], [node1['y_km'], node2['y_km']],
                'gray', alpha=0.3, linewidth=0.5)
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')
    plt.title('Network Topology')
    plt.grid(True)


    plt.subplot(224)
    labels = ['Accepted', 'Rejected']
    sizes = [len(df_accepted), len(df_rejected)]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
    plt.axis('equal')
    plt.title('Order Acceptance')

    plt.tight_layout()
    plt.savefig('uav_path_optimization_results.png')
    plt.show()


def run_offline_optimization(df_trips_prepared, df_links, shortest_paths):
    """
    运行离线优化算法
    """
    print("Running Offline Optimization...")
    start_time = time.time()

    optimization_result = solve_offline_optimization_with_gurobi(
        df_trips_prepared,
        df_links,
        shortest_paths,
        uav_speed=45.0,
        time_step_min=1,
        capacity=3
    )

    runtime = time.time() - start_time

    if optimization_result and optimization_result['status'] == GRB.OPTIMAL:
        return {
            'total_welfare': optimization_result['objective'],
            'acceptance_rate': optimization_result['acceptance_rate'],
            'runtime': runtime,
            'accepted_orders': optimization_result['accepted_orders']
        }
    else:
        print("Offline optimization failed.")
        return {
            'total_welfare': 0,
            'acceptance_rate': 0,
            'runtime': runtime,
            'accepted_orders': []
        }


def compare_all_algorithms(df_trips_prepared, df_links, shortest_paths):

    results = {}

    # 运行离线优化
    offline_results = run_offline_optimization(df_trips_prepared, df_links, shortest_paths)
    results['Offline Optimal'] = offline_results

    # 运行在线算法
    online_results = compare_online_algorithms(df_trips_prepared, df_links, shortest_paths)
    results.update(online_results)

    return results


def plot_full_comparison_results(results):

    algorithms = list(results.keys())
    welfares = [results[algo]['total_welfare'] for algo in algorithms]
    acceptance_rates = [results[algo]['acceptance_rate'] * 100 for algo in algorithms]  # 转换为百分比
    runtimes = [results[algo]['runtime'] for algo in algorithms]


    offline_welfare = results['Offline Optimal']['total_welfare'] if 'Offline Optimal' in results else 1
    welfare_ratios = [w / offline_welfare * 100 for w in welfares]

    plt.figure(figsize=(18, 10))


    plt.subplot(221)
    bars = plt.bar(algorithms, welfares, color=['red', 'blue', 'green', 'orange'])
    plt.title('Total Welfare Comparison')
    plt.ylabel('Total Welfare')
    plt.grid(axis='y')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom')


    plt.subplot(222)
    bars = plt.bar(algorithms, acceptance_rates, color=['red', 'blue', 'green', 'orange'])
    plt.title('Acceptance Rate Comparison')
    plt.ylabel('Acceptance Rate (%)')
    plt.grid(axis='y')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')


    plt.subplot(223)
    bars = plt.bar(algorithms, runtimes, color=['red', 'blue', 'green', 'orange'])
    plt.title('Runtime Comparison')
    plt.ylabel('Runtime (seconds)')
    plt.grid(axis='y')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom')


    plt.subplot(224)
    bars = plt.bar(algorithms, welfare_ratios, color=['red', 'blue', 'green', 'orange'])
    plt.title('Performance Relative to Offline Optimal')
    plt.ylabel('Percentage of Offline Optimal (%)')
    plt.grid(axis='y')
    plt.axhline(y=100, color='r', linestyle='--', alpha=0.5)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('full_algorithm_comparison.png')
    plt.show()


    results_df = pd.DataFrame({
        'Algorithm': algorithms,
        'Total Welfare': welfares,
        'Acceptance Rate (%)': acceptance_rates,
        'Runtime (s)': runtimes,
        'Relative to Offline (%)': welfare_ratios
    })

    print("\nAlgorithm Performance Comparison:")
    print(results_df)


    if 'Offline Optimal' in results and results['Offline Optimal']['total_welfare'] > 0:
        for algo in ['Primal-Dual', 'Greedy', 'FCFS']:
            if algo in results:
                ratio = results[algo]['total_welfare'] / results['Offline Optimal']['total_welfare']
                print(f"Competitive ratio for {algo}: {ratio:.4f}")


if __name__ == "__main__":

    csv_path = "all_waybill_info_meituan_0322.csv"
    df_orders = read_meituan_orders(csv_path)
    df_nodes, df_trips = build_node_and_trip_tables(df_orders)


    node_stats, duplicated_nodes = check_node_occurrences(df_trips)


    if len(duplicated_nodes) > 0:
        top_10_percent = duplicated_nodes[duplicated_nodes['total_count'] >=
                                        np.percentile(duplicated_nodes['total_count'], 80)]
        top_node_ids = top_10_percent['node_id'].tolist()
    else:

        top_node_ids = node_stats['node_id'].tolist()

    df_top_trips = df_trips[
        (df_trips['pickup_id'].isin(top_node_ids)) &
        (df_trips['dropoff_id'].isin(top_node_ids))
    ].reset_index(drop=True)


    filtered_df = df_top_trips[
        df_top_trips['pickup_id'].isin(top_node_ids) &
        df_top_trips['dropoff_id'].isin(top_node_ids)
    ]


    dist_dict = {
        (row['pickup_id'], row['dropoff_id']): row['distance_km']
        for _, row in filtered_df.iterrows()
    }


    dist_dict = make_symmetric(dist_dict)

    start_time = time.time()

    df_nodes_top, df_links, df_trips_prepared, node_id_to_idx, idx_to_node_id, adj_matrix = prepare_data(
        df_top_trips, top_node_ids, df_nodes, dist_dict, radius_km=3.0
    )

    end_time = time.time()

    print("Data preparation time:", (end_time-start_time)/60)

    start_time = time.time()

    shortest_paths = find_shortest_paths(df_links, node_id_to_idx, idx_to_node_id, dist_dict )

    end_time = time.time()

    print("Calculating shortest path time:", (end_time - start_time)/60) # 6 hours

    start_time = time.time()


    all_results = compare_all_algorithms(df_trips_prepared, df_links, shortest_paths)

    end_time = time.time()

    print("Performance comparison duration:", (end_time - start_time)/60)


    plot_full_comparison_results(all_results)


    results_df = pd.DataFrame({
        'Algorithm': list(all_results.keys()),
        'Total Welfare': [all_results[algo]['total_welfare'] for algo in all_results],
        'Acceptance Rate (%)': [all_results[algo]['acceptance_rate'] * 100 for algo in all_results],
        'Runtime (s)': [all_results[algo]['runtime'] for algo in all_results]
    })
    results_df.to_csv('algorithm_comparison_results_15.csv', index=False)



    import pandas as pd
    import numpy as np
    from pyproj import Transformer





    from matplotlib.patches import FancyArrowPatch


    def convert_coords_to_km(df_nodes):
        """
        Convert encrypted integer lat/lon (micro-degrees) into planar (x_km, y_km)
        coordinates using Web Mercator projection.
        """
        df_nodes["lon"] = df_nodes["x"] / 1e6
        df_nodes["lat"] = df_nodes["y"] / 1e6

        transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
        x_m, y_m = transformer.transform(df_nodes["lon"].values, df_nodes["lat"].values)

        df_nodes["x_km"] = x_m / 1000
        df_nodes["y_km"] = y_m / 1000
        return df_nodes[["node_id", "x_km", "y_km"]]



    oid_idxs = np.where(filtered_df.columns.values == 'order_id')[0]
    if len(oid_idxs) == 0:
        raise KeyError("df_trips_prepared 不存在 'order_id' 列")

    oid_tp = filtered_df.iloc[:, oid_idxs[0]]


    ids = pd.to_numeric(oid_tp, errors='coerce').dropna().astype('int64').unique()


    df_orders_ids = pd.to_numeric(df_orders['order_id'], errors='coerce')
    df_orders_sub = df_orders[df_orders_ids.isin(ids)].copy()




    sender_nodes = pd.DataFrame({
        "node_id": df.index,
        "x": df["sender_lng"],
        "y": df["sender_lat"]
    })
    recipient_nodes = pd.DataFrame({
        "node_id": df.index,
        "x": df["recipient_lng"],
        "y": df["recipient_lat"]
    })
    grab_nodes = pd.DataFrame({
        "node_id": df.index,
        "x": df["grab_lng"],
        "y": df["grab_lat"]
    })


    sender_xy = convert_coords_to_km(sender_nodes).set_index("node_id")
    recipient_xy = convert_coords_to_km(recipient_nodes).set_index("node_id")
    grab_xy = convert_coords_to_km(grab_nodes).set_index("node_id")


    df[["sender_x","sender_y"]] = sender_xy[["x_km","y_km"]]
    df[["recipient_x","recipient_y"]] = recipient_xy[["x_km","y_km"]]
    df[["grab_x","grab_y"]] = grab_xy[["x_km","y_km"]]


    df["dist_gs_km"] = np.hypot(df["grab_x"] - df["sender_x"], df["grab_y"] - df["sender_y"])
    df["dist_sr_km"] = np.hypot(df["sender_x"] - df["recipient_x"], df["sender_y"] - df["recipient_y"])
    df["distance_km"] = df["dist_gs_km"] + df["dist_sr_km"]


    UAV_SPEED_KMPH = 45.0
    df["uav_duration_min"] = (df["distance_km"] / UAV_SPEED_KMPH) * 60.0
    df["human_duration_min"] = (df["arrive_time"] - df["fetch_time"]) / 60.0


    df["human_on_time"] = df["arrive_time"] <= df["estimate_arrived_time"]
    df["uav_arrive_time"] = df["fetch_time"] + df["uav_duration_min"] * 60.0
    df["uav_on_time"] = df["uav_arrive_time"] <= df["estimate_arrived_time"]


    print(df[[
        "order_id","distance_km","uav_duration_min","human_duration_min",
        "human_on_time","uav_on_time"
    ]].head())




    def classify_trip(d):
        if d < 1:
            return "short (<1 km)"
        elif d <= 3:
            return "medium (1-3 km)"
        else:
            return "long (>3 km)"

    df["trip_class"] = df["distance_km"].apply(classify_trip)

    df["human_delay_min"] = np.maximum(0, (df["arrive_time"] - df["estimate_arrived_time"]) / 60)
    df["uav_delay_min"] = np.maximum(0, ((df["fetch_time"] + df["uav_duration_min"] * 60) - df["estimate_arrived_time"]) / 60)


    df["human_early_min"] = np.maximum(0, (df["estimate_arrived_time"] - df["arrive_time"]) / 60)
    df["uav_early_min"] = np.maximum(0, (df["estimate_arrived_time"] - (df["fetch_time"] + df["uav_duration_min"] * 60)) / 60)


    df["human_on_time"] = (df["arrive_time"] <= df["estimate_arrived_time"]).astype(int)
    df["uav_on_time"] = ((df["fetch_time"] + df["uav_duration_min"] * 60) <= df["estimate_arrived_time"]).astype(int)


    summary = df.groupby("trip_class").agg(
        total_orders=("order_id", "count"),
        avg_human_delay_min=("human_delay_min", "mean"),
        avg_uav_delay_min=("uav_delay_min", "mean"),
        avg_human_early_min=("human_early_min", "mean"),
        avg_uav_early_min=("uav_early_min", "mean"),
        human_on_time_ratio=("human_on_time", "mean"),
        uav_on_time_ratio=("uav_on_time", "mean")
    ).reset_index()


    summary["human_on_time_ratio"] *= 100
    summary["uav_on_time_ratio"] *= 100


    summary["delay_gap_%"] = ((summary["avg_human_delay_min"] - summary["avg_uav_delay_min"])
                              / summary["avg_human_delay_min"].replace(0, np.nan) * 100)

    summary["early_gap_%"] = ((summary["avg_uav_early_min"] - summary["avg_human_early_min"])
                              / summary["avg_human_early_min"].replace(0, np.nan) * 100)

    summary["on_time_gap_%"] = summary["uav_on_time_ratio"] - summary["human_on_time_ratio"]





    order = ["short (<1 km)", "medium (1-3 km)", "long (>3 km)"]
    summary = summary.set_index("trip_class").reindex(order).reset_index()

    summary.to_csv("Topfiftypercent.csv", index=False, encoding="utf-8-sig")





    def add_increase_arrow_strict(ax, human_vals, uav_vals, x, width,
                                  x_at="center",
                                  fontsize=20, line_lw=3,
                                  head_px=34,
                                  head_len_ratio=0.02,
                                  head_overshoot_ratio=0.003,
                                  label_offset_ratio=0.03):


        human_vals = np.asarray(human_vals, dtype=float)
        uav_vals   = np.asarray(uav_vals, dtype=float)

        ymin, ymax = ax.get_ylim()
        if not np.isfinite(ymin) or not np.isfinite(ymax) or ymax <= ymin:
            ymax = float(np.nanmax([human_vals.max(), uav_vals.max()]) * 1.1)
            ymin = 0.0
            ax.set_ylim(ymin, ymax)

        yspan   = ax.get_ylim()[1] - ax.get_ylim()[0]
        ypad    = label_offset_ratio   * yspan
        headlen = head_len_ratio       * yspan
        overs   = head_overshoot_ratio * yspan

        def xi_at(i):
            if x_at == "left":
                return x[i] - width/2
            elif x_at == "right":
                return x[i] + width/2
            return x[i]

        for i in range(len(x)):
            h = float(human_vals[i])
            u = float(uav_vals[i])
            xi = xi_at(i)

            # （
            ax.plot([x[i] - width/2, x[i] + width/2], [h, u],
                    marker="o", color="black", linestyle="--", linewidth=2, zorder=3)

            #
            if np.isfinite(h) and h > 0 and np.isfinite(u):
                pct = (u - h) / h * 100.0
                label = f"+{abs(pct):.1f}%"
            else:
                label = "—"

            #
            ax.text(xi, u + ypad, label, color="red", fontsize=fontsize,
                    ha="center", va="bottom", fontweight="bold", zorder=8)

            #
            y0, y1 = (u, h) if u < h else (h, u)
            ax.vlines(xi, y0, y1, colors="red", linewidth=line_lw, zorder=5)

            #
            if u < h:
                tail_y = u + headlen
                end_y  = u - overs
            else:
                tail_y = u - headlen
                end_y  = u + overs

            head = FancyArrowPatch(
                (xi, tail_y), (xi, end_y),
                arrowstyle='-|>',
                mutation_scale=head_px,
                color='red', linewidth=0, zorder=7
            )
            head.set_clip_on(False)
            ax.add_patch(head)


    def add_delay_arrow_strict(ax, human_vals, uav_vals, x, width,
                               x_at="center", fontsize=20, line_lw=3,
                               head_px=34, head_len_ratio=0.02, head_overshoot_ratio=0.003,
                               label_offset_ratio=0.03):

        human_vals = np.asarray(human_vals, dtype=float)
        uav_vals   = np.asarray(uav_vals, dtype=float)

        ymin, ymax = ax.get_ylim()
        if not np.isfinite(ymin) or not np.isfinite(ymax) or ymax <= ymin:
            ymax = float(np.nanmax([human_vals.max(), uav_vals.max()]) * 1.1)
            ymin = 0.0
            ax.set_ylim(ymin, ymax)

        yspan   = ax.get_ylim()[1] - ax.get_ylim()[0]
        ypad    = label_offset_ratio   * yspan
        headlen = head_len_ratio       * yspan
        overs   = head_overshoot_ratio * yspan

        def xi_at(i):
            if x_at == "left":
                return x[i] - width/2
            elif x_at == "right":
                return x[i] + width/2
            return x[i]  # center

        for i in range(len(x)):
            h = float(human_vals[i])
            u = float(uav_vals[i])
            xi = xi_at(i)


            ax.plot([x[i] - width/2, x[i] + width/2], [h, u],
                    marker="o", color="black", linestyle="--", linewidth=2, zorder=3)


            if np.isfinite(h) and h > 0 and np.isfinite(u):
                pct = (h - u) / h * 100.0
                label = f"-{abs(pct):.1f}%"
            else:
                label = "—"
            ax.text(xi, h + ypad, label, color="red", fontsize=fontsize,
                    ha="center", va="bottom", fontweight="bold", zorder=8)


            y0, y1 = (u, h) if u < h else (h, u)
            ax.vlines(xi, y0, y1, colors="red", linewidth=line_lw, zorder=5)


            if u < h:

                tail_y = u + headlen
                end_y  = u - overs
            else:

                tail_y = u - headlen
                end_y  = u + overs

            head = FancyArrowPatch(
                (xi, tail_y), (xi, end_y),
                arrowstyle='-|>', mutation_scale=head_px,
                color='red', linewidth=0, zorder=7
            )
            head.set_clip_on(False)
            ax.add_patch(head)




    human_color = "burlywood"
    uav_color = "skyblue"
    alpha_val   = 0.45

    '''
    Meituan dataset, on-demand delivery
    '''


    # ========= average delay =========

    x = np.arange(len(summary))
    width = 0.36

    fig1, ax1 = plt.subplots(figsize=(12, 8))

    # left human，right UAV
    human_delay = summary["avg_human_delay_min"].values
    uav_delay   = summary["avg_uav_delay_min"].values

    ax1.bar(x - width/2, human_delay, width, label="Human courier",
            color=human_color, alpha=alpha_val  )
    ax1.bar(x + width/2, uav_delay,   width, label="UAV courier",
            color=uav_color, alpha=alpha_val)


    ax1.set_xticks(x)
    ax1.set_xticklabels(summary["trip_class"])


    ax1.set_ylabel("Average delay (min)", fontsize=25)
    ax1.set_xlabel("Trip class", fontsize=25)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    ax1.tick_params(axis="both", labelsize=25)
    ax1.set_ylim(0, 1.2)


    ax1.margins(x=0.02)


    add_delay_arrow_strict(
        ax1,
        summary["avg_human_delay_min"].values,
        summary["avg_uav_delay_min"].values,
        x, width,
        x_at="center",
        head_px=34,
        head_len_ratio=0.02,
        head_overshoot_ratio=0.008
    )



    ax1.legend(fontsize=25)
    fig1.tight_layout()





    fig2, ax2 = plt.subplots(figsize=(12, 8))

    human_early = summary["avg_human_early_min"].values
    uav_early   = summary["avg_uav_early_min"].values

    ax2.bar(x - width/2, human_early, width, label="Human courier",
            color=human_color, alpha=alpha_val)
    ax2.bar(x + width/2, uav_early,   width, label="UAV courier",
            color=uav_color, alpha=alpha_val)

    ax2.set_xticks(x)
    ax2.set_xticklabels(summary["trip_class"])

    ax2.set_ylabel("Average early delivery (minutes)", fontsize=25)
    ax2.set_xlabel("Trip class", fontsize=25)
    ax2.legend(fontsize = 25, loc = 'upper left')
    ax2.grid(axis="y", linestyle="--", alpha=0.6)
    ax2.tick_params(axis="both", labelsize=25)
    ax2.margins(x=0.02)
    ax2.set_ylim(0, 30)
    ax2.set_yticks(range(0, 31, 5))




    add_increase_arrow_strict(
        ax2,
        summary["avg_human_early_min"].values,
        summary["avg_uav_early_min"].values,
        x, width,
        x_at="center",
        head_px=34,
        head_len_ratio=0.02,
        head_overshoot_ratio=0.0065,
        label_offset_ratio=0.03
    )


    fig2.tight_layout()



    fig3, ax3 = plt.subplots(figsize=(12, 8))

    human_ratio = summary["human_on_time_ratio"].values
    uav_ratio   = summary["uav_on_time_ratio"].values

    ax3.bar(x - width/2, human_ratio, width, label="Human courier",
            color=human_color, alpha=alpha_val)
    ax3.bar(x + width/2, uav_ratio,   width, label="UAV courier",
            color=uav_color, alpha=alpha_val)

    ax3.set_xticks(x)
    ax3.set_xticklabels(summary["trip_class"])

    ax3.set_ylabel("On-time ratio (%)", fontsize=25)
    ax3.set_xlabel("Trip class", fontsize=25)
    ax3.legend(fontsize = 25, loc = 'upper left')
    ax3.grid(axis="y", linestyle="--", alpha=0.6)
    ax3.tick_params(axis="both", labelsize=25)
    ax3.margins(x=0.02)
    ax3.set_ylim(0, 140)
    ax3.set_yticks(range(0, 101, 20))




    add_increase_arrow_strict(
        ax3,
        summary["human_on_time_ratio"].values,
        summary["uav_on_time_ratio"].values,
        x, width,
        x_at="center",
        head_px=34,
        head_len_ratio=0.02,
        head_overshoot_ratio=0.0065,
        label_offset_ratio=0.03
    )

    fig3.tight_layout()

    plt.show()





import numpy as np
import pandas as pd

def build_summary_table(df_orders_sub: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:


    df = df_orders_sub.copy()

    # ---- distances (km) ----
    df["dist_gs_km"] = np.hypot(df["grab_x"] - df["sender_x"], df["grab_y"] - df["sender_y"])
    df["dist_sr_km"] = np.hypot(df["sender_x"] - df["recipient_x"], df["sender_y"] - df["recipient_y"])
    df["distance_km"] = df["dist_gs_km"] + df["dist_sr_km"]

    # ---- durations (min) ----
    UAV_SPEED_KMPH = 45.0
    df["uav_duration_min"] = (df["distance_km"] / UAV_SPEED_KMPH) * 60.0
    df["human_duration_min"] = (df["arrive_time"] - df["fetch_time"]) / 60.0

    # ---- on-time indicators ----
    df["uav_arrive_time"] = df["fetch_time"] + df["uav_duration_min"] * 60.0
    df["human_on_time"] = (df["arrive_time"] <= df["estimate_arrived_time"]).astype(int)
    df["uav_on_time"] = (df["uav_arrive_time"] <= df["estimate_arrived_time"]).astype(int)

    # ---- delay / early (min) ----
    df["human_delay_min"] = np.maximum(0, (df["arrive_time"] - df["estimate_arrived_time"]) / 60.0)
    df["uav_delay_min"]   = np.maximum(0, (df["uav_arrive_time"] - df["estimate_arrived_time"]) / 60.0)

    df["human_early_min"] = np.maximum(0, (df["estimate_arrived_time"] - df["arrive_time"]) / 60.0)
    df["uav_early_min"]   = np.maximum(0, (df["estimate_arrived_time"] - df["uav_arrive_time"]) / 60.0)

    # ---- trip class ----
    def classify_trip(d):
        if d < 1:
            return "short (<1 km)"
        elif d <= 3:
            return "medium (1-3 km)"
        return "long (>3 km)"

    df["trip_class"] = df["distance_km"].apply(classify_trip)

    # ---- class-level summary ----
    summary = df.groupby("trip_class", as_index=False).agg(
        total_orders=("order_id", "count"),
        avg_human_delay_min=("human_delay_min", "mean"),
        avg_uav_delay_min=("uav_delay_min", "mean"),
        avg_human_early_min=("human_early_min", "mean"),
        avg_uav_early_min=("uav_early_min", "mean"),
        human_on_time_ratio=("human_on_time", "mean"),
        uav_on_time_ratio=("uav_on_time", "mean"),
    )

    # ratios -> percent
    summary["human_on_time_ratio"] *= 100.0
    summary["uav_on_time_ratio"] *= 100.0

    # gaps
    summary["delay_gap_%"] = np.where(
        summary["avg_human_delay_min"] > 0,
        (summary["avg_human_delay_min"] - summary["avg_uav_delay_min"]) / summary["avg_human_delay_min"] * 100.0,
        np.nan
    )
    summary["early_gap_%"] = np.where(
        summary["avg_human_early_min"] > 0,
        (summary["avg_uav_early_min"] - summary["avg_human_early_min"]) / summary["avg_human_early_min"] * 100.0,
        np.nan
    )
    summary["on_time_gap_pp"] = summary["uav_on_time_ratio"] - summary["human_on_time_ratio"]

    # consistent ordering
    order = ["short (<1 km)", "medium (1-3 km)", "long (>3 km)"]
    summary["trip_class"] = pd.Categorical(summary["trip_class"], categories=order, ordered=True)
    summary = summary.sort_values("trip_class").reset_index(drop=True)

    # ---- build display table (nice column names) ----
    display = summary[[
        "trip_class", "total_orders",
        "avg_human_delay_min", "avg_uav_delay_min", "delay_gap_%",
        "avg_human_early_min", "avg_uav_early_min", "early_gap_%",
        "human_on_time_ratio", "uav_on_time_ratio", "on_time_gap_pp"
    ]].copy()

    display = display.rename(columns={
        "trip_class": "Trip class",
        "total_orders": "Orders",
        "avg_human_delay_min": "Delay (min) - Human",
        "avg_uav_delay_min": "Delay (min) - UAV",
        "delay_gap_%": "Delay gap (%)",
        "avg_human_early_min": "Early (min) - Human",
        "avg_uav_early_min": "Early (min) - UAV",
        "early_gap_%": "Early gap (%)",
        "human_on_time_ratio": "On-time (%) - Human",
        "uav_on_time_ratio": "On-time (%) - UAV",
        "on_time_gap_pp": "On-time gap (pp)",
    })

    # Optional: rounding for prettier printing
    num_cols = [c for c in display.columns if c not in ["Trip class", "Orders"]]
    display[num_cols] = display[num_cols].astype(float).round(2)
    display["Orders"] = pd.to_numeric(display["Orders"], errors="coerce").astype("Int64")

    # order-level table (optional to inspect)
    order_table = df[[
        "order_id", "trip_class", "distance_km",
        "human_duration_min", "uav_duration_min",
        "human_delay_min", "uav_delay_min",
        "human_early_min", "uav_early_min",
        "human_on_time", "uav_on_time"
    ]].copy()

    # rounding
    for c in ["distance_km", "human_duration_min", "uav_duration_min",
              "human_delay_min", "uav_delay_min", "human_early_min", "uav_early_min"]:
        order_table[c] = order_table[c].astype(float).round(3)

    return order_table, display



order_table, summary_table = build_summary_table(df)

print("\n=== Class-level summary table (for paper) ===")
print(summary_table.to_string(index=False))

print("\n=== Order-level table preview (first 10 rows) ===")
print(order_table.head(10).to_string(index=False))

# Save if needed
summary_table.to_csv("summary_table_for_paper.csv", index=False, encoding="utf-8-sig")
order_table.to_csv("order_level_metrics.csv", index=False, encoding="utf-8-sig")
print("\nSaved: summary_table_for_paper.csv, order_level_metrics.csv")
