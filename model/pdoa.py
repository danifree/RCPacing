import math
import copy

import random
import numpy as np
from scipy import stats, special

from data.data_loader import *


class Func:
    def __init__(self, params):
        self.params = params


class Params:
    def __init__(self):
        self.T = 50  # dual update total times

        # PDOA
        self.beta = 0.1

        # learning-rate
        self.eta_expo_decay = 0.85
        self.step_size_max = 0.005

        # max budget ratio
        self.linear_inc = True
        self.max_budget_ratio = [1.2, 2.0]


class PDOA:
    def __init__(self, supply, demand, edge):
        # load data
        self.params = Params()
        self.func = Func(self.params)
        self.alpha_dual = None
        self.t = 0
        self.supply = supply
        self.demand = demand
        self.edge = edge
        self.demand_num = self.demand.demand_num
        self.supply_num = self.supply.supply_num
        self.budgets = self.demand.demand_mat.reshape(self.demand_num)

    def alloc_all(self, n=None):
        self._init_df()
        self._init_before_start()
        batch_size = self.supply_num // self.params.T
        for i in range(self.params.T):
            self.t += 1
            supply_ctr = self.edge.ctr[i * batch_size: (i + 1) * batch_size]
            supply_gamma = self.edge.gamma[i *
                                           batch_size: (i + 1) * batch_size]
            self._update_budget()
            self._alloc_t(supply_ctr, supply_gamma)
            self._metric()
            self._update_dual()
            if n is not None and i >= n:
                break

    def _alloc_t(self, mat_ctr, mat_gamma):
        alpha_avg = np.sum(self.weight[:, np.newaxis] * self.alpha_dual, axis=0)
        for i in range(mat_ctr.shape[0]):
            ctr_list, gamma_list = mat_ctr[i], mat_gamma[i]
            # price
            budget_remain = self.budget_max > 0.0
            bid = (ctr_list - alpha_avg)
            pr = bid > 0.0
            price = (ctr_list - alpha_avg)
            bid_final = gamma_list * pr * price * budget_remain
            idx_max = np.argmax(bid_final)
            if bid_final[idx_max] > 0:
                self.cost_last[idx_max] += 1
                self.cost_accu[idx_max] += 1
                self.budget_max[idx_max] -= 1
                self.ctr_total[idx_max] += ctr_list[idx_max]
        # gradient M x 1
        grad = self.cost_expect - self.cost_last
        grad = grad[:, np.newaxis]
        # surrogate loss N
        sur_loss = np.matmul(alpha_avg[np.newaxis, :] - self.alpha_dual, grad).squeeze()
        # update the weights
        self.weight = self.weight * np.exp(-self.params.beta * sur_loss)
        self.weight = self.weight / np.sum(self.weight)

    def _update_dual(self):
        if self.params.T - self.t in (0, 1):
            self.alpha_dual *= 0.95
            return
        if self.t % 5 == 0:
            self.eta = self.eta * self.params.eta_expo_decay
        # compute gradient
        eta = self.eta1 if self.t == 1 else self.eta
        g = self.cost_expect - self.cost_last
        g = g[np.newaxis, :] * eta[:, np.newaxis]

        # static clip
        g = np.clip(g, -self.params.step_size_max, self.params.step_size_max)

        alpha_update = self.alpha_dual - g

        alpha_update = np.clip(alpha_update, 0., 1.)
        self.alpha_dual = alpha_update

    def _update_budget(self):
        lb, ub = self.params.max_budget_ratio
        if self.params.linear_inc:
            max_budget_ratio = (ub - lb) / self.params.T * self.t + lb
        else:
            max_budget_ratio = ub
        self.budget_left = np.clip(self.budgets - self.cost_accu, 0, None)
        self.cost_expect = self.budget_left * \
                           1.0 / (self.params.T - self.t + 1)
        self.budget_max = np.minimum.reduce(
            [self.cost_expect * max_budget_ratio, self.budget_left])
        self.cost_last *= 0

    def _metric(self):
        #       smooth loss
        self.smooth_loss += (self.cost_last - self.cost_total_avg) ** 2
        smooth_loss_mean = round(np.mean((self.smooth_loss / self.t) ** 0.5), 3)
        eta = round(self.eta[0], 3)
        finish_ratio = round(np.sum(self.cost_last) /
                             (np.sum(self.cost_expect) + 1e-6), 3)
        alpha_dual_avg = round(np.mean(self.alpha_dual), 3)
        finish_total = round(np.sum(self.cost_accu) / np.sum(self.budgets), 5)
        ctr_all = round(np.sum(self.ctr_total), 0)
        ctr_avg = round(np.sum(self.ctr_total) /
                        (np.sum(self.cost_accu) + 1), 5)
        cost_last = round(np.sum(self.cost_last), 1)

        loss_this_round = np.round(
            np.mean((self.cost_last - self.cost_total_avg) ** 2), 1)
        print(f"round {self.t}:")
        print(
            f"finish ratio {finish_ratio},  eta {eta}, finish total {finish_total}, alpha_dual_avg {alpha_dual_avg},"
            f"ctr_avg {ctr_avg}, ctr_all {ctr_all}, smooth_loss {smooth_loss_mean}, loss_this {loss_this_round}, "
            f"cost_last {cost_last}")

    def _init_df(self):
        self.gamma = self.edge.gamma
        self.ctr_org = self.edge.ctr

        self.nb_expert = int(np.ceil(0.5 * np.log2(1 + 4 * self.params.T / 7)))
        self.cost_accu, self.cost_last, self.budget_left, self.cost_expect, \
        self.budget_max, self.ctr_total, self.smooth_loss = (
            np.ones(self.demand_num, dtype=np.float64) for _
            in range(7))
        self.alpha_dual = np.zeros((self.nb_expert, self.demand_num), dtype=np.float64)
        self.eta, self.eta1, self.weight = (
            np.zeros(self.nb_expert, dtype=np.float64) for _
            in range(3))

    def _init_before_start(self):
        self.cost_total_avg = self.budgets * 1.0 / self.params.T
        self.alpha_dual *= 0
        self.cost_accu *= 0
        self.cost_last *= 0
        self.ctr_total *= 0
        self.smooth_loss *= 0

        for i in range(1, self.nb_expert+1):
            self.eta1[i-1] = 2 ** (i - 1) / max(np.max(self.budgets), self.supply_num /
                                               self.params.T - np.max(self.budgets))
            self.weight[i-1] = (self.nb_expert + 1) / (i * (i + 1) * self.nb_expert)

        self.eta = self.eta1 * 0.5


if __name__ == '__main__':
    supply_file, demand_file = ['./data/supply_demand_data.txt'] * 2
    supply = Supply(supply_file)
    demand = Demand(demand_file)
    edge = Edge(supply_file, demand, supply)
    op = PDOA(supply, demand, edge)
    op.alloc_all(100)
