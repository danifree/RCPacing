"""
load demand and supply data
"""

import numpy as np

class Supply:
    def __init__(self, supply_file):
        self.satisfy_demand = []
        self.satisfy_demand_num = []
        self.supply_num = 0
        with open(supply_file, 'r') as f:
            for line_id, line in enumerate(f):
                line = line.strip()
                if line.startswith('#') or line_id == 0:
                    continue
                self.supply_num += 1
                arrive_time, ad_info = line.split('|')

                ad_info_list = ad_info.split(';')
                demand_list = []
                for demand_score_pair in ad_info_list:
                    demand_id, ctr = demand_score_pair.split(':')
                    demand_list.append(int(demand_id))
                self.satisfy_demand.append(demand_list)
                self.satisfy_demand_num.append(len(demand_list))
        f.close()
        print("load supply finished, supply num:", self.supply_num)

    def get_supply_num(self):
        return self.supply_num

    def get_satisfy_demand(self, supply_id):
        return self.satisfy_demand[supply_id]


# budget_pv|id0:budget0;id1:budget1 ...
class Demand:
    def __init__(self, demand_file):
        self.demand = []  # 各demand的预算
        self.demand_num = 0
        self.demand_sum = 0
        self.target_supply = {}  # 与各demand相匹配的supply
        with open(demand_file, 'r') as f:
            demand_info = f.readline().strip().split('|')[-1]
            demand_info = demand_info.split(';')
            for id_demand in demand_info:
                demand_id, budget = id_demand.split(':')
                budget = int(budget)
                self.demand.append(budget)
                self.demand_num += 1
                self.demand_sum += budget
        f.close()

        self.demand_mat = np.array(self.demand, dtype=np.float64).reshape(1, self.demand_num)
        print("load demand finished, demand num:", self.demand_num, "demand sum:", self.demand_sum)

    def get_demand_num(self):
        return self.demand_num


# demand_id1:ctr1;demand_id2:ctr2;...
class Edge:
    def __init__(self, supply_file, demand, supply, scale=2.5*5e5):
        self.edge = {}
        supply_num = supply.supply_num
        demand_num = demand.demand_num
        self.gamma = np.zeros((supply_num, demand_num), dtype=int)
        self.ctr = np.zeros((supply_num, demand_num), dtype=np.float64)
        self.edge_num = 0

        with open(supply_file, 'r') as f:
            for line_id, line in enumerate(f):
                line = line.strip()
                if line.startswith('#') or line_id == 0:
                    continue
                supply_id = line_id - 1
                arrive_time, ad_info = line.split('|')
                ad_info_list = ad_info.split(';')
                for demand_score_pair in ad_info_list:
                    self.edge_num += 1
                    demand_id, ctr = demand_score_pair.split(':')
                    demand_id = int(demand_id)
                    self.ctr[supply_id, demand_id] = float(ctr) / scale
                    self.gamma[supply_id, demand_id] = 1
        f.close()

        print("load edge   finished, edge_num", self.edge_num)

if __name__ == '__main__':
    demand = Demand('./data/supply_demand_data.txt')
    supply = Supply('./data/supply_demand_data.txt')
    edge = Edge('./data/supply_demand_data.txt', demand, supply)
