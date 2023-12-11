import math
import copy

import random
import numpy as np
from scipy import stats, special

from data.data_loader import *


class Func:
    def __init__(self, params):
        self.params = params

    def layer_division(self, data):
        """
        :param data:  np.narray
        :return:  layers
        """
        # sort
        sorted_data = sorted(list(data.reshape(-1, )))

        # remove 0 values
        #non_zero = 0
        #while non_zero < len(sorted_data) and sorted_data[non_zero] == 0:
        #    non_zero += 1
        #sorted_data = sorted_data[non_zero:]

        supply_per_layer = len(sorted_data) // self.params.n_layers

        boarders = sorted_data[::supply_per_layer][:self.params.n_layers]
        boarders[0] = 0
        boarders.append(1)

        layers = []
        for i in range(self.params.n_layers):
            layers.append((boarders[i], boarders[i + 1]))

        return layers

    @staticmethod
    def search_layer(ctr, layers):
        """
        search layer according to the ctr

        :param ctr: float
        :param layers: list

        :return: layer_idx
        """
        i, j = 0, len(layers)-1
        while i <= j:
            m = (i + j) // 2
            if layers[m][1] <= ctr:
                i = m + 1
            elif layers[m][0] > ctr:
                j = m - 1
            else:
                return m

    def get_ptr_layer(self, layer_rate, layers, ctr_list, gamma_list):
        """
        :param layer_rate: np.narray
        :param layers: list
        :param ctr_list: np.narray
        :return: ptr, layer
        """
        ptr = np.zeros(len(ctr_list), dtype=np.float32)
        layer = np.zeros(len(ctr_list), dtype=np.uint8)
        for idx, ctr in enumerate(ctr_list):
            if gamma_list[idx] == 0:
                continue
            layer_idx = self.search_layer(ctr, layers)
            _ptr = layer_rate[layer_idx, idx]
            ptr[idx] = _ptr
            layer[idx] = layer_idx

        return ptr, layer

    def expect_cur_cost(self, budget, cost_accu, t):
        """
        compute expected cost at current time period

        :param budget: np.narray
        :param cost_accu: np.narray
        :param t: int

        :return: cur_cost
        """
        budget_left_real = np.clip(budget - cost_accu, 0, None)
        budget_left_expe = budget / self.params.T * (self.params.T - t)

        cur_cost = budget / self.params.T + (budget_left_real - budget_left_expe) / (self.params.T - t)

        return cur_cost


class Params:
    def __init__(self):
        self.T = 50  # dual update total times

        # learning-rate
        self.eta = 2e-5

        # smart pacing
        self.n_layers = 100
        self.trial_rates = 0.001
        self.layer_update_switch = False

        # max budget ratio
        self.linear_inc = True
        self.max_budget_ratio = [1.2, 4.0]


class SmartPacing:
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
            logs_ctr = self.edge.ctr[max(0, i - 1) * batch_size:i * batch_size]
            logs_gamma = self.edge.gamma[max(0, i - 1) * batch_size:i * batch_size]
            self._update_layer(logs_ctr, logs_gamma)
            supply_ctr = self.edge.ctr[i * batch_size: (i + 1) * batch_size]
            supply_gamma = self.edge.gamma[i *
                                           batch_size: (i + 1) * batch_size]
            self._update_budget()
            self._alloc_t(supply_ctr, supply_gamma)
            self._metric()
            self._update_dual()
            if self.t < self.params.T:
                self._update_ptr()
            if n is not None and i >= n:
                break

    def _alloc_t(self, mat_ctr, mat_gamma):
        for i in range(mat_ctr.shape[0]):
            ctr_list, gamma_list = mat_ctr[i], mat_gamma[i]
            # obtain ptr
            ptr_t, layer = self.func.get_ptr_layer(self.layer_rate, self.layers, ctr_list, gamma_list)
            # close pacing in the last periods
            # if self.params.T - self.t <= 1:
            #    ptr_t = np.clip(ptr_t * 2 ** (self.t - self.params.T + 2), 0., 1.)
            ptr_res = np.random.rand(self.demand_num) <= ptr_t
            # price
            budget_remain = self.budget_max > 0.0
            bid = (ctr_list - self.alpha_dual)
            pr = bid > 0.0
            price = (ctr_list - self.alpha_dual)
            bid_final = gamma_list * ptr_res * pr * price * budget_remain
            idx_max = np.argmax(bid_final)
            if bid_final[idx_max] > 0:
                self.layer_cost[layer[idx_max], idx_max] += 1
                self.cost_accu[idx_max] += 1
                self.budget_max[idx_max] -= 1
                self.ctr_total[idx_max] += ctr_list[idx_max]

            keep = gamma_list * ptr_res * budget_remain
            if keep[idx_max] > 0:
                self.cost_virtual[idx_max] += 1

    def _update_dual(self):
        if self.params.T - self.t in (0, 1):
            self.alpha_dual *= 0.95
            return
        # compute gradient
        eta = self.params.eta
        cost_last = self.cost_virtual

        g = (self.cost_expect - cost_last) * eta

        alpha_update = self.alpha_dual - g

        alpha_update = np.clip(alpha_update, 0.001, 0.999)
        self.alpha_dual = alpha_update

    def _update_layer(self, data, gamma):
        if self.t == 1 or not self.params.layer_update_switch:
            return
        self.layers = self.func.layer_division(data[gamma == 1])

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
        self.layer_cost *= 0
        self.cost_virtual *= 0

    def _update_ptr(self):
        rate_last = self.layer_rate.copy()
        cost_last = self.layer_cost.copy()

        rate_now = rate_last.copy()

        # difference between B_t and C_{t-1}
        cur_cost_expect = self.func.expect_cur_cost(self.budgets, self.cost_accu, self.t)
        last_cost_real = np.sum(cost_last, axis=0)
        residual = cur_cost_expect - last_cost_real

        for d_idx in range(self.demand_num):
            if self.cost_accu[d_idx] >= self.budgets[d_idx]:
                continue
            res = residual[d_idx]
            if res == 0:
                continue
            elif res > 0:
                # get the highest layer
                l_max = len(rate_last) - 1
                # get the lowest layer
                l_nonzero = 0
                while l_nonzero <= l_max and rate_last[l_nonzero, d_idx] == 0:
                    l_nonzero += 1
                assert l_nonzero <= l_max
                # adjust layer by layer in a top-down manner
                for l in range(l_max, l_nonzero-1, -1):
                    rate_now[l, d_idx] = rate_last[l, d_idx] * (1. + res / (cost_last[l, d_idx] + 1e-3))
                    rate_now[l, d_idx] = min([1., rate_now[l, d_idx]])
                    res -= (cost_last[l, d_idx] + 1e-3) * (rate_now[l, d_idx] - rate_last[l, d_idx]) \
                           / (rate_last[l, d_idx] + 1e-3)
                # trial layer
                if l_nonzero > 0 and rate_now[l_nonzero, d_idx] > self.params.trial_rates:
                    rate_now[l_nonzero-1, d_idx] = self.params.trial_rates
            else:
                # get the highest layer
                l_max = len(rate_last) - 1
                # get the lowest layer
                l_nonzero = 0
                while l_nonzero <= l_max and rate_last[l_nonzero, d_idx] == 0:
                    l_nonzero += 1
                assert l_nonzero <= l_max
                for l in range(l_nonzero, l_max+1):
                    rate_now[l, d_idx] = rate_last[l, d_idx] * (1. + res / (cost_last[l, d_idx] + 1e-3))
                    rate_now[l, d_idx] = max([0., rate_now[l, d_idx]])
                    res -= (cost_last[l, d_idx] + 1e-3) * (rate_now[l, d_idx] - rate_last[l, d_idx]) \
                           / (rate_last[l, d_idx] + 1e-3)
                    # trial layer
                    if res >= 0:
                        if l > 0 and rate_now[l, d_idx] > self.params.trial_rates:
                            rate_now[l-1, d_idx] = self.params.trial_rates
                        break

        # update ptr
        self.layer_rate = rate_now.copy()

    def _metric(self):
        #       smooth loss
        self.smooth_loss += (np.sum(self.layer_cost, axis=0) - self.cost_total_avg) ** 2
        smooth_loss_mean = round(np.mean((self.smooth_loss / self.t) ** 0.5), 3)
        eta = round(self.params.eta, 3)
        finish_ratio = round(np.sum(self.layer_cost) /
                             (np.sum(self.cost_expect) + 1e-6), 3)
        alpha_dual_avg = round(np.mean(self.alpha_dual), 3)
        finish_total = round(np.sum(self.cost_accu) / np.sum(self.budgets), 5)
        ctr_all = round(np.sum(self.ctr_total), 0)
        ctr_avg = round(np.sum(self.ctr_total) /
                        (np.sum(self.cost_accu) + 1), 5)
        layer_cost = round(np.sum(self.layer_cost), 1)

        loss_this_round = np.round(
            np.mean((np.sum(self.layer_cost, axis=0) - self.cost_total_avg) ** 2), 1)
        print(f"round {self.t}:")
        print(
            f"finish ratio {finish_ratio},  eta {eta}, finish total {finish_total}, alpha_dual_avg {alpha_dual_avg},"
            f"ctr_avg {ctr_avg}, ctr_all {ctr_all}, smooth_loss {smooth_loss_mean}, loss_this {loss_this_round}, "
            f"cost_last {layer_cost}")

    def _init_df(self):
        self.gamma = self.edge.gamma
        self.ctr_org = self.edge.ctr
        self.alpha_dual, self.cost_accu, self.budget_left, self.cost_expect, \
        self.budget_max, self.ctr_total, self.smooth_loss, self.cost_virtual = (
            np.ones(self.demand_num, dtype=np.float64) for _
            in range(8))

        self.layer_rate, self.layer_cost = (
            np.ones((self.params.n_layers, self.demand_num), dtype=np.float64) for _
            in range(2))

    def _init_before_start(self):
        self.cost_total_avg = self.budgets * 1.0 / self.params.T
        self.alpha_dual *= 0
        self.cost_accu *= 0
        self.ctr_total *= 0
        self.smooth_loss *= 0
        self.layer_rate *= (1 / self.params.n_layers)
        self.layer_cost *= 0
        self.cost_virtual *= 0

        # init layer global
        stats_init = min(5 * 10000, self.ctr_org.shape[0])
        ctr = self.ctr_org[0:stats_init]
        gamma = self.gamma[0:stats_init]
        self.layers = self.func.layer_division(ctr[gamma == 1])


if __name__ == '__main__':
    supply_file, demand_file = ['./data/supply_demand_data.txt'] * 2
    supply = Supply(supply_file)
    demand = Demand(demand_file)
    edge = Edge(supply_file, demand, supply)
    op = SmartPacing(supply, demand, edge)
    op.alloc_all(100)
