


import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time


# =========================================================
# Utility: earliest feasible start with waiting (queueing)
# =========================================================
def earliest_start_with_wait(u, z, C, time_horizon, max_wait=None):

   if not u['path']:
       return u['tau_dep'], [], 0


   base_abs = min(t for (_, t) in u['path'])
   rel_path = [(l, t_abs - base_abs) for (l, t_abs) in u['path']]
   dur = max((r for (_, r) in rel_path), default=0) + 1

   earliest = int(u['tau_dep'])
   latest = int(min(time_horizon - dur, u['tau_arr'] + u['delta'] - dur + 1))
   if max_wait is not None:
       latest = min(latest, earliest + max_wait)
   if latest < earliest:
       return None, None, None

   for tau in range(earliest, latest + 1):
       shifted = [(l, tau + r) for (l, r) in rel_path]
       feasible = all(z.get((l, t), 0) < C.get((l, t), 0) for (l, t) in shifted)
       if feasible:
           return tau, shifted, tau - earliest
   return None, None, None


# =========================================================
# Offline optimal via Gurobi (relative-time, feasible-start set)
# =========================================================
def solve_offline_with_gurobi(all_users, C, links, time_horizon):

   try:
       import gurobipy as gp
       from gurobipy import GRB
   except Exception as e:
       print(f"[Gurobi not available] {e}")
       return 0.0, 0.0, 0.0, 0.0

   try:
       rel_paths = {}
       S = {}
       for u in all_users:
           i = u['id']
           tau_dep = u['tau_dep']
           tau_arr = u['tau_arr']
           delta   = u['delta']
           # 相对路径（基于 path 的绝对时刻）
           rel_paths[i] = [(l, t_abs - tau_dep) for (l, t_abs) in u['path']]
           dur = (max((r for (_, r) in rel_paths[i]), default=-1) + 1) if rel_paths[i] else 1
           earliest_start = int(tau_dep)
           latest_start   = int(min(time_horizon - dur, tau_arr + delta - dur + 1))
           S[i] = list(range(earliest_start, latest_start + 1)) if latest_start >= earliest_start else []

       model = gp.Model()
       model.setParam("OutputFlag", 0)

       x = {}
       for u in all_users:
           i = u['id']
           for tau in S[i]:
               x[i, tau] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{tau}")
       model.update()


       for u in all_users:
           i = u['id']
           if S[i]:
               model.addConstr(gp.quicksum(x[i, tau] for tau in S[i]) <= 1)


       for l in links:
           for t in range(time_horizon):
               expr = gp.LinExpr()
               for u in all_users:
                   i = u['id']
                   for (lp, rel) in rel_paths[i]:
                       if lp != l:
                           continue
                       for tau in S[i]:
                           if tau + rel == t:
                               expr += x[i, tau]
               model.addConstr(expr <= C[l, t])

       model.setObjective(
           gp.quicksum(u['bid'] * x[u['id'], tau] for u in all_users for tau in S[u['id']]),
           GRB.MAXIMIZE
       )

       t0 = time.time()
       model.optimize()
       t1 = time.time()


       delays = []
       accepted = 0
       for u in all_users:
           i = u['id']
           chose = None
           for tau in S[i]:
               var = x.get((i, tau))
               if var is not None and var.X > 0.5:
                   chose = tau
                   break
           if chose is not None:
               accepted += 1
               delays.append(chose - u['tau_dep'])

       obj_val = float(model.ObjVal) if model.status == GRB.OPTIMAL else 0.0
       acc_ratio = accepted / max(1, len(all_users))
       avg_delay = float(np.mean(delays)) if delays else 0.0
       runtime = t1 - t0
       return obj_val, acc_ratio, runtime, avg_delay

   except Exception as e:
       print(f"[Gurobi solve error] {e}")
       return 0.0, 0.0, 0.0, 0.0


# =========================================================
# Adversarial user arrival generator (two-phase) + randomization
# =========================================================
def generate_adversarial_users(num_users, time_horizon, links,
                              T_split=7,
                              spike_gap=1,
                              spike_len=8,
                              crit_ratio=0.35,
                              low_bid=(10, 1200),
                              high_bid=(300_000, 5_000_000),
                              low_path_len=(3, 6),
                              high_path_len=(1, 2),
                              early_share=0.72,
                              seed=12,
                              arrival_randomization=None,   # None | 'poisson' | 'uniform'
                              preserve_totals=True):

   rng = np.random.default_rng(seed)

   num_crit = max(1, int(len(links) * crit_ratio))
   crit_links = links[:num_crit]

   n_early = int(num_users * early_share)
   n_late  = max(0, num_users - n_early)


   T_split = max(2, T_split)
   early_ts = np.arange(0, min(T_split, time_horizon))
   spike_start = min(T_split + spike_gap, time_horizon - 1)
   spike_end = min(spike_start + spike_len, time_horizon)
   if spike_end <= spike_start:
       spike_end = min(time_horizon, spike_start + 1)
   late_ts = np.arange(spike_start, spike_end)
   if len(late_ts) == 0:
       late_ts = np.array([time_horizon - 1])


   def split_counts(T_slots, total, mode):
       if T_slots <= 0 or total <= 0:
           return np.zeros(0, dtype=int)
       if mode is None:
           base = total // T_slots
           cnts = np.full(T_slots, base, dtype=int)
           for i in range(total % T_slots):
               cnts[i] += 1
           return cnts
       elif mode == 'uniform':
           probs = np.full(T_slots, 1.0 / T_slots)
           return rng.multinomial(total, probs)
       elif mode == 'poisson':
           lam = total / T_slots
           rates = rng.poisson(lam, size=T_slots).astype(float)
           if rates.sum() <= 0:
               probs = np.full(T_slots, 1.0 / T_slots)
           else:
               probs = rates / rates.sum()
           return rng.multinomial(total, probs)
       else:
           raise ValueError(f"Unknown arrival_randomization: {mode}")

   early_counts = split_counts(len(early_ts), n_early, arrival_randomization)
   late_counts  = split_counts(len(late_ts),  n_late,  arrival_randomization)


   users, uid = [], 0

   def make_path(dep_t, path_len, allowed_links):
       times = list(range(dep_t, min(dep_t + path_len, time_horizon)))
       if not times:
           times = [min(dep_t, time_horizon - 1)]
       return [(rng.choice(allowed_links), tt) for tt in times]

   # early phase
   for t, k in zip(early_ts, early_counts):
       for _ in range(int(k)):
           bid = rng.integers(low_bid[0], low_bid[1] + 1)
           pl  = rng.integers(low_path_len[0], low_path_len[1] + 1)
           dep = int(t)
           path = make_path(dep, int(pl), crit_links)
           arr = dep + max(1, len(path)) - 1
           delta = rng.integers(0, 3)
           users.append({
               'id': uid, 'o': rng.choice(links), 'd': rng.choice(links),
               'bid': int(bid), 'tau_dep': dep, 'tau_arr': min(arr, time_horizon - 1),
               'delta': int(delta), 'path': path
           })
           uid += 1

   # late spike
   for t, k in zip(late_ts, late_counts):
       for _ in range(int(k)):
           bid = rng.integers(high_bid[0], high_bid[1] + 1)
           pl  = rng.integers(high_path_len[0], high_path_len[1] + 1)
           dep = int(t)
           path = make_path(dep, int(pl), crit_links)
           arr = dep + max(1, len(path)) - 1
           delta = rng.integers(0, 2)
           users.append({
               'id': uid, 'o': rng.choice(links), 'd': rng.choice(links),
               'bid': int(bid), 'tau_dep': dep, 'tau_arr': min(arr, time_horizon - 1),
               'delta': int(delta), 'path': path
           })
           uid += 1

   users = users[:num_users]
   users.sort(key=lambda u: u['tau_dep'])
   for i, u in enumerate(users):
       u['id'] = i
   return users


# =========================================================
# Main experiment runner (allow waiting + delay stats)
# =========================================================
def run_experiment(
   num_runs=10,
   num_users=1000,
   time_horizon=20,
   links=None,
   initial_price=0.1,
   price_cap=50.0,
   use_adversarial_arrivals=True,
   random_seed=12,
   arrival_randomization=None,

   beta_lookahead=0.8,
   margin_factor=1.05,
   base_window=3,
   window_scale=6,
   window_bias=4,
   greedy_window=2,
   fcfs_window=2,
   max_wait=None
):
   if links is None:
       links = ['L' + str(i) for i in range(60)]

   random.seed(random_seed)
   np.random.seed(random_seed)

   results = {
       'Rolling Welfare': [], 'Rolling Accepted': [], 'Rolling Time': [], 'Rolling Delay': [],
       'FCFS Welfare': [], 'FCFS Accepted': [], 'FCFS Time': [], 'FCFS Delay': [],
       'Offline Welfare': [], 'Offline Accepted': [], 'Offline CPU time': [], 'Offline Delay': [],
       'Greedy Welfare': [], 'Greedy Accepted': [], 'Greedy CPU time': [], 'Greedy Delay': []
   }

   full_start_time = time.time()

   for run in range(num_runs):
       # capacity per link-time
       C = {(l, t): random.randint(3, 5) for l in links for t in range(time_horizon)}
       # PD price & usage trackers
       pi = {(l, t): initial_price for (l, t) in C}
       z_rolling = {(l, t): 0 for (l, t) in C}

       # arrivals
       if use_adversarial_arrivals:
           all_users = generate_adversarial_users(
               num_users=num_users,
               time_horizon=time_horizon,
               links=links,
               T_split=7,
               spike_gap=1,
               spike_len=8,
               crit_ratio=0.35,
               low_bid=(10, 1200),
               high_bid=(300_000, 5_000_000),
               low_path_len=(3, 6),
               high_path_len=(1, 2),
               early_share=0.72,
               seed=random_seed + run,
               arrival_randomization=arrival_randomization,  # <- 选择 'poisson' / 'uniform' / None
               preserve_totals=True
           )
       else:
           # simple stationary Poisson fallback
           arrival_rate = num_users / (time_horizon - 5)
           all_users = []
           user_id = 0
           for t in range(time_horizon - 5):
               if user_id >= num_users:
                   break
               arrivals_this_step = np.random.poisson(arrival_rate)
               for _ in range(arrivals_this_step):
                   if user_id >= num_users:
                       break
                   o_i = random.choice(links); d_i = random.choice(links)
                   tau_dep = t; tau_arr = tau_dep + random.randint(1, 4)
                   delta_i = random.randint(0, 3); path_len = random.randint(1, 4)
                   full_time_range = [(l, ts) for l in links for ts in range(tau_dep, min(tau_arr + delta_i + 1, time_horizon))]
                   if len(full_time_range) >= path_len:
                       path = sorted(random.sample(full_time_range, path_len), key=lambda x: x[1])
                       b_i = random.randint(10, 1_000_000)
                       all_users.append({'id': user_id, 'o': o_i, 'd': d_i, 'bid': b_i,
                                         'tau_dep': tau_dep, 'tau_arr': tau_arr,
                                         'delta': delta_i, 'path': path})
                       user_id += 1

       # -------------------------------------
       # PD with lookahead + waiting
       # -------------------------------------
       start_time = time.time()
       accepted_rolling = []
       accepted_ids_rolling = set()
       delays_pd = []

       for t in range(0, time_horizon):

           future_load = sum(len([u for u in all_users if t <= u['tau_dep'] < t+i]) for i in range(1, time_horizon - t + 1))
           rolling_window = max(base_window,
                                min(time_horizon - t, int(window_scale * future_load / max(1, len(all_users))) + window_bias))

           window_users = [u for u in all_users if t <= u['tau_dep'] < t + rolling_window and u['id'] not in accepted_ids_rolling]
           if not window_users:
               continue


           demand_forecast = {}
           for u in window_users:
               tau_s, shifted_s, _ = earliest_start_with_wait(u, z_rolling, C, time_horizon, max_wait)
               path_eval = shifted_s if shifted_s is not None else u['path']
               for (l, tt) in path_eval:
                   if t <= tt < t + rolling_window:
                       demand_forecast[(l, tt)] = demand_forecast.get((l, tt), 0) + 1


           def efficiency(u):
               tau_s, shifted, _ = earliest_start_with_wait(u, z_rolling, C, time_horizon, max_wait)
               if tau_s is None:
                   return -1e18
               est = 0.0
               cong = 0.0
               for (l, tt) in shifted:
                   df = demand_forecast.get((l, tt), 0) / max(1, C.get((l, tt), 1))
                   lookahead_price = pi.get((l, tt), initial_price) * (1.0 + beta_lookahead * df)
                   est += min(lookahead_price, price_cap)
                   cong += z_rolling.get((l, tt), 0) / max(1, C.get((l, tt), 1))
               avg_cong = cong / max(1, len(shifted))
               return u['bid'] / ((1.0 + est) * (1.0 + avg_cong))

           window_users.sort(key=efficiency, reverse=True)

           for u in window_users:
               if u['id'] in accepted_ids_rolling:
                   continue

               tau_s, shifted, wait_steps = earliest_start_with_wait(u, z_rolling, C, time_horizon, max_wait)
               if tau_s is None:
                   continue


               p_i = 0.0
               for (l, tt) in shifted:
                   df = demand_forecast.get((l, tt), 0) / max(1, C.get((l, tt), 1))
                   lookahead_price = pi.get((l, tt), initial_price) * (1.0 + beta_lookahead * df)
                   p_i += min(lookahead_price, price_cap)
               p_i = min(p_i, price_cap)

              
               if u['bid'] >= margin_factor * p_i:
                   accepted_rolling.append((u, p_i))
                   accepted_ids_rolling.add(u['id'])
                   delays_pd.append(wait_steps)
                   for (l, tt) in shifted:
                       z_rolling[(l, tt)] += 1

                       B_i = max(sum(C.get((lx, tx), 1) for (lx, tx) in shifted), 1)
                       base_increase = u['bid'] / (B_i * C[(l, tt)])
                       congestion_factor = z_rolling[(l, tt)] / C[(l, tt)]
                       pi_increase = base_increase * (1 + 2 * congestion_factor**2)
                       pi[(l, tt)] = min(pi[(l, tt)] * (1 + 0.2 * np.log1p(pi_increase)), price_cap)

       end_time = time.time()
       results['Rolling Welfare'].append(sum(u['bid'] for u, _ in accepted_rolling))
       results['Rolling Accepted'].append(len(accepted_rolling) / max(1, len(all_users)))
       results['Rolling Time'].append(end_time - start_time)
       results['Rolling Delay'].append(float(np.mean(delays_pd)) if delays_pd else 0.0)

       # -------------------------------------
       # FCFS with waiting (short window, no lookahead)
       # -------------------------------------
       accepted_fcfs = []
       accepted_ids_fcfs = set()
       z_fcfs = {(l, t): 0 for (l, t) in C}
       delays_fcfs = []
       start_time = time.time()

       for t2 in range(0, time_horizon, fcfs_window):
           window_users = [u for u in all_users if t2 <= u['tau_dep'] < t2 + fcfs_window and u['id'] not in accepted_ids_fcfs]
           window_users.sort(key=lambda u: u['tau_dep'])
           for u in window_users:
               tau, shifted, wait_steps = earliest_start_with_wait(u, z_fcfs, C, time_horizon, max_wait)
               if tau is None:
                   continue
               accepted_fcfs.append((u, 0))
               accepted_ids_fcfs.add(u['id'])
               delays_fcfs.append(wait_steps)
               for (l, tt) in shifted:
                   z_fcfs[(l, tt)] += 1

       end_time = time.time()
       results['FCFS Welfare'].append(sum(u['bid'] for u, _ in accepted_fcfs))
       results['FCFS Accepted'].append(len(accepted_fcfs) / max(1, len(all_users)))
       results['FCFS Time'].append(end_time - start_time)
       results['FCFS Delay'].append(float(np.mean(delays_fcfs)) if delays_fcfs else 0.0)

       # -------------------------------------
       # Greedy-by-bid with waiting (short window, no lookahead)
       # -------------------------------------
       accepted_greedy = []
       accepted_ids_greedy = set()
       z_greedy = {(l, t): 0 for (l, t) in C}
       delays_greedy = []
       start_time = time.time()

       for t3 in range(0, time_horizon, greedy_window):
           window_users = [u for u in all_users if t3 <= u['tau_dep'] < t3 + greedy_window and u['id'] not in accepted_ids_greedy]
           window_users.sort(key=lambda u: u['bid'], reverse=True)
           for u in window_users:
               tau, shifted, wait_steps = earliest_start_with_wait(u, z_greedy, C, time_horizon, max_wait)
               if tau is None:
                   continue
               accepted_greedy.append((u, 0))
               accepted_ids_greedy.add(u['id'])
               delays_greedy.append(wait_steps)
               for (l, tt) in shifted:
                   z_greedy[(l, tt)] += 1

       end_time = time.time()
       results['Greedy Welfare'].append(sum(u['bid'] for u, _ in accepted_greedy))
       results['Greedy Accepted'].append(len(accepted_greedy) / max(1, len(all_users)))
       results['Greedy CPU time'].append(end_time - start_time)
       results['Greedy Delay'].append(float(np.mean(delays_greedy)) if delays_greedy else 0.0)

       # -------------------------------------
       # Offline optimal (Gurobi MILP) with avg delay
       # -------------------------------------
       obj_val, accept_ratio, cpu_time, avg_delay = solve_offline_with_gurobi(all_users, C, links, time_horizon)
       results['Offline Welfare'].append(obj_val)
       results['Offline Accepted'].append(accept_ratio)
       results['Offline CPU time'].append(cpu_time)
       results['Offline Delay'].append(avg_delay)

   # ----- Welfare gaps (already added before) -----
   gap_rolling = 100 * (np.array(results['Offline Welfare']) - np.array(results['Rolling Welfare'])) \
                   / np.maximum(1e-9, np.array(results['Offline Welfare']))
   gap_fcfs = 100 * (np.array(results['Offline Welfare']) - np.array(results['FCFS Welfare'])) \
                   / np.maximum(1e-9, np.array(results['Offline Welfare']))
   gap_greedy = 100 * (np.array(results['Offline Welfare']) - np.array(results['Greedy Welfare'])) \
                   / np.maximum(1e-9, np.array(results['Offline Welfare']))

   min_gap_rolling, max_gap_rolling = np.min(gap_rolling), np.max(gap_rolling)
   min_gap_fcfs, max_gap_fcfs = np.min(gap_fcfs), np.max(gap_fcfs)
   min_gap_greedy, max_gap_greedy = np.min(gap_greedy), np.max(gap_greedy)

   # ----- Accepted-rate gaps (new) -----
   acc_gap_rolling = 100 * (np.array(results['Offline Accepted']) - np.array(results['Rolling Accepted'])) \
                       / np.maximum(1e-9, np.array(results['Offline Accepted']))
   acc_gap_fcfs = 100 * (np.array(results['Offline Accepted']) - np.array(results['FCFS Accepted'])) \
                       / np.maximum(1e-9, np.array(results['Offline Accepted']))
   acc_gap_greedy = 100 * (np.array(results['Offline Accepted']) - np.array(results['Greedy Accepted'])) \
                       / np.maximum(1e-9, np.array(results['Offline Accepted']))

   min_acc_gap_rolling, max_acc_gap_rolling = np.min(acc_gap_rolling), np.max(acc_gap_rolling)
   min_acc_gap_fcfs, max_acc_gap_fcfs = np.min(acc_gap_fcfs), np.max(acc_gap_fcfs)
   min_acc_gap_greedy, max_acc_gap_greedy = np.min(acc_gap_greedy), np.max(acc_gap_greedy)


   # aggregate
   df_final = pd.DataFrame({
       'Algorithm': ['Primal-Dual (lookahead)', 'FCFS', 'Greedy (bid, short window)', 'Offline Optimal'],
       'Avg Social Welfare': [
           np.mean(results['Rolling Welfare']),
           np.mean(results['FCFS Welfare']),
           np.mean(results['Greedy Welfare']),
           np.mean(results['Offline Welfare']),
       ],
       'Std Social Welfare': [
           np.std(results['Rolling Welfare']),
           np.std(results['FCFS Welfare']),
           np.std(results['Greedy Welfare']),
           np.std(results['Offline Welfare']),
       ],
       'Avg Accepted Users': [
           np.mean(results['Rolling Accepted']),
           np.mean(results['FCFS Accepted']),
           np.mean(results['Greedy Accepted']),
           np.mean(results['Offline Accepted']),
       ],
       'Avg Runtime (s)': [
           np.mean(results['Rolling Time']),
           np.mean(results['FCFS Time']),
           np.mean(results['Greedy CPU time']),
           np.mean(results['Offline CPU time']),
       ],
       'Avg Delay (steps)': [
           np.mean(results['Rolling Delay']),
           np.mean(results['FCFS Delay']),
           np.mean(results['Greedy Delay']),
           np.mean(results['Offline Delay']),
       ],
       'Gap to Offline (%)': [
           100 * np.mean((np.array(results['Offline Welfare']) - np.array(results['Rolling Welfare'])) / np.maximum(1e-9, np.array(results['Offline Welfare']))),
           100 * np.mean((np.array(results['Offline Welfare']) - np.array(results['FCFS Welfare'])) / np.maximum(1e-9, np.array(results['Offline Welfare']))),
           100 * np.mean((np.array(results['Offline Welfare']) - np.array(results['Greedy Welfare'])) / np.maximum(1e-9, np.array(results['Offline Welfare']))),
           0.0
       ],
       'Gap in CPU time (s)': [
           np.mean((np.array(results['Offline CPU time']) - np.array(results['Rolling Time']))),
           np.mean((np.array(results['Offline CPU time']) - np.array(results['FCFS Time']))),
           np.mean((np.array(results['Offline CPU time']) - np.array(results['Greedy CPU time']))),
           0.0
       ],
       'Gap in Accepted rate (%)': [
           100 * np.mean((np.array(results['Offline Accepted']) - np.array(results['Rolling Accepted'])) / np.maximum(1e-9, np.array(results['Offline Accepted']))),
           100 * np.mean((np.array(results['Offline Accepted']) - np.array(results['FCFS Accepted'])) / np.maximum(1e-9, np.array(results['Offline Accepted']))),
           100 * np.mean((np.array(results['Offline Accepted']) - np.array(results['Greedy Accepted'])) / np.maximum(1e-9, np.array(results['Offline Accepted']))),
           0.0
       ],
       'Min Gap to Offline (%)': [
       min_gap_rolling, min_gap_fcfs, min_gap_greedy, 0.0
       ],
       'Max Gap to Offline (%)': [
           max_gap_rolling, max_gap_fcfs, max_gap_greedy, 0.0
       ],


       'Min Gap in Accepted rate (%)': [
           min_acc_gap_rolling,
           min_acc_gap_fcfs,
           min_acc_gap_greedy,
           0.0
       ],
       'Max Gap in Accepted rate (%)': [
           max_acc_gap_rolling,
           max_acc_gap_fcfs,
           max_acc_gap_greedy,
           0.0
       ]


   })

   return df_final, results


def visualize(df_final):
   ax = df_final.set_index('Algorithm')[['Avg Social Welfare']].plot(kind='bar', legend=False)
   ax.set_ylabel("Average Social Welfare")
   ax.set_title("Algorithm Comparison: PD (lookahead) vs FCFS vs Greedy vs Offline")
   plt.tight_layout()
   plt.show()

   ax2 = df_final.set_index('Algorithm')[['Avg Accepted Users']].plot(kind='bar', legend=False)
   ax2.set_ylabel("Average Accepted Users (ratio)")
   ax2.set_title("Acceptance Rate Comparison")
   plt.tight_layout()
   plt.show()

   ax3 = df_final.set_index('Algorithm')[['Avg Delay (steps)']].plot(kind='bar', legend=False)
   ax3.set_ylabel("Average Delay (steps)")
   ax3.set_title("Average Waiting Time (steps)")
   plt.tight_layout()
   plt.show()



if __name__ == "__main__":
   import numpy as np
   import matplotlib.pyplot as plt
   import matplotlib as mpl
   import pandas as pd


   all_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]



   arrival_modes = ["poisson", "uniform"]


   results_per_mode = {}
   gaps_per_mode = {}
   acc_gaps_per_mode = {}

   for ARR_MODE in arrival_modes:
       print(f"\n===== Running experiments with arrival_randomization = {ARR_MODE} =====")
       results_per_size = {}
       gaps = {}
       acc_gaps = {}

       for n_users in all_sizes:
           print(f"  -> size = {n_users}")
           df_final, results = run_experiment(
               num_runs=2000,
               num_users=n_users,
               time_horizon=20,
               links=['L' + str(i) for i in range(60)],
               initial_price=0.1,
               price_cap=50.0,
               use_adversarial_arrivals=True,
               random_seed=15,
               arrival_randomization=ARR_MODE,
               beta_lookahead=0.8,
               margin_factor=1.05,
               base_window=3,
               window_scale=6,
               window_bias=4,
               greedy_window=2,
               fcfs_window=2,
               max_wait=None
           )

           results_per_size[n_users] = results

           # ----- welfare gap -----
           offline_w = np.array(results['Offline Welfare'])
           roll_w    = np.array(results['Rolling Welfare'])
           fcfs_w    = np.array(results['FCFS Welfare'])
           greedy_w  = np.array(results['Greedy Welfare'])

           gap_rolling = 100.0 * (offline_w - roll_w)   / np.maximum(1e-9, offline_w)
           gap_fcfs    = 100.0 * (offline_w - fcfs_w)   / np.maximum(1e-9, offline_w)
           gap_greedy  = 100.0 * (offline_w - greedy_w) / np.maximum(1e-9, offline_w)

           gaps[n_users] = {
               "Primal–Dual": gap_rolling,
               "FCFS":        gap_fcfs,
               "Greedy":      gap_greedy
           }

           # ----- accepted-rate gap -----
           off_acc  = np.array(results['Offline Accepted'])
           pd_acc   = np.array(results['Rolling Accepted'])
           fcfs_acc = np.array(results['FCFS Accepted'])
           g_acc    = np.array(results['Greedy Accepted'])

           gap_pd_acc   = 100.0 * (off_acc - pd_acc)   / np.maximum(1e-9, off_acc)
           gap_fcfs_acc = 100.0 * (off_acc - fcfs_acc) / np.maximum(1e-9, off_acc)
           gap_g_acc    = 100.0 * (off_acc - g_acc)    / np.maximum(1e-9, off_acc)

           acc_gaps[n_users] = {
               "Primal–Dual": gap_pd_acc,
               "FCFS":        gap_fcfs_acc,
               "Greedy":      gap_g_acc
           }


           print(f"     mean welfare gap PD/FCFS/Greedy = "
                 f"{gap_rolling.mean():.3f}%, {gap_fcfs.mean():.3f}%, {gap_greedy.mean():.3f}%")
           print(f"     mean accept  gap PD/FCFS/Greedy = "
                 f"{gap_pd_acc.mean():.3f}%, {gap_fcfs_acc.mean():.3f}%, {gap_g_acc.mean():.3f}%")

       results_per_mode[ARR_MODE] = results_per_size
       gaps_per_mode[ARR_MODE] = gaps
       acc_gaps_per_mode[ARR_MODE] = acc_gaps


   alg_order = ["Primal–Dual", "FCFS", "Greedy", "Offline"]
   rows = []

   for mode in arrival_modes:
       gaps_mode = gaps_per_mode[mode]
       acc_mode  = acc_gaps_per_mode[mode]

       for n in all_sizes:
           for alg in alg_order:
               if alg == "Offline":
                   # Offline: gap 恒为 0
                   w_mean = w_min = w_max = 0.0
                   a_mean = a_min = a_max = 0.0
               else:
                   w_samples = gaps_mode[n][alg]
                   a_samples = acc_mode[n][alg]

                   w_mean = float(np.nanmean(w_samples))
                   w_min  = float(np.nanmin(w_samples))
                   w_max  = float(np.nanmax(w_samples))
                   a_mean = float(np.nanmean(a_samples))
                   a_min  = float(np.nanmin(a_samples))
                   a_max  = float(np.nanmax(a_samples))

               rows.append({
                   "Arrival mode": mode,
                   "Instance size": n,
                   "Algorithm": alg,
                   "Welfare gap mean (%)": round(w_mean, 4),
                   "Welfare gap min (%)":  round(w_min, 4),
                   "Welfare gap max (%)":  round(w_max, 4),
                   "Accepted-rate gap mean (%)": round(a_mean, 4),
                   "Accepted-rate gap min (%)":  round(a_min, 4),
                   "Accepted-rate gap max (%)":  round(a_max, 4),
               })

   df_all = pd.DataFrame(rows)


   alg_rank = {a: i for i, a in enumerate(alg_order)}
   df_all = df_all.sort_values(
       by=["Arrival mode", "Instance size", df_all["Algorithm"].map(alg_rank)]
   ).reset_index(drop=True)

   print(df_all)


   csv_path = "gaps_summary_poisson_vs_uniform.csv"
   df_all.to_csv(csv_path, index=False)
   print(f"\nSaved summary CSV to: {csv_path}")
