import math
import copy

import random
import numpy as np
from scipy import stats, special

from data.data_loader import *


class Func:
    def __init__(self, params):
        self.params = params

    @staticmethod
    def solve_max(coef, y):
        sum = 0.0
        sum_k = 0.0
        coef = sorted(coef, key=lambda t: t[0])
        x0 = coef[0][0]
        for entry in coef:
            sum += (entry[0] - x0) * sum_k
            if sum + 1e-8 >= y:
                if sum_k == 0.0:
                    if sum > y:
                        print('max error 1')
                        return 10000
                else:
                    return entry[0] - (sum - y) / sum_k
            x0 = entry[0]
            if entry[1] == 0:
                sum_k += entry[2]
            else:
                sum_k -= entry[2]
        print('max error 2')
        return 20000

    # calculate beta_i by solving sum_j(xij)=1 for each supply node i
    def calculate_beta_i(self, ctr_mat, gamma_mat, alpha_j, theta_j):
        # lower bound
        a1 = alpha_j - self.params.obj_lambda * ctr_mat - self.params.obj_weight - self.params.obj_priority
        # upper bound
        a2 = alpha_j - self.params.obj_lambda * ctr_mat - self.params.obj_weight + self.params.obj_priority \
             * (1.0 / theta_j - 1)
        # slope
        b = theta_j / self.params.obj_priority
        beta_array = []
        for i in range(ctr_mat.shape[0]):
            coef = []
            cntt = 0
            total_upper_bound = 0.0
            beta_i, _beta = 0.0, 0.0
            for j in range(alpha_j.shape[0]):
                if gamma_mat[i, j] <= 0 or b[j] == 0:
                    continue
                coef.append((a1[i, j], 0, b[j]))
                coef.append((a2[i, j], 1, b[j]))
                if a1[i, j] > a2[i, j] + self.params.error:
                    cntt += 1
                total_upper_bound += 1.0
            if cntt > 0:
                print('coef length = ' + str(len(coef)) + ', error cnt = ' + str(cntt))
                print(coef)
            if total_upper_bound < 1.0 + self.params.error:
                beta_i = 0.0
            else:
                result = self.solve_max(coef, 1.0)
                if result == 10000 or result == 20000:
                    beta_i = 0.0
                else:
                    if result > 0:
                        beta_i = 0.0
                    else:
                        if abs(result + _beta) < self.params.error:
                            beta_i = _beta
                        else:
                            beta_i = - 1.0 * result
            beta_array.append(beta_i)
        ret_beta = np.array(beta_array, dtype=np.float32)

        return ret_beta

    def calculate_x_ij(self, ctr_list, gamma_list, alpha_j, beta_i, theta_j):
        # x_ij = max(0, theta * (w_j + (1 + lambda_j * ctr - alpha - beta) / vj))
        x_ij = np.minimum(1, np.maximum(0, theta_j * (1 + (
                self.params.obj_weight + self.params.obj_lambda * ctr_list - alpha_j - beta_i) /
                                                           self.params.obj_priority))) * gamma_list

        return x_ij

    @staticmethod
    def search(x, data):
        assert 0 <= x <= 1

        lb, rb = 0, len(data)-1
        while lb < len(data) and data[lb] == 0:
            lb += 1
        while rb > 0 and data[rb] == data[rb-1]:
            rb -= 1

        if x <= data[lb]:
            return lb
        elif x >= data[rb]:
            return rb

        i, j = lb + 1, rb
        while i <= j:
            m = (i + j) // 2
            if data[m-1] <= x < data[m]:
                return m
            elif x < data[m-1]:
                j = m - 1
            else:
                i = m + 1


class Params:
    def __init__(self):
        self.T = 50  # dual update total times

        # learning-rate
        self.eta = 0.01

        # max budget ratio
        self.linear_inc = True
        self.max_budget_ratio = [1.2, 2]

        # auaf
        self.error = 1e-10
        self.obj_lambda = 100
        self.obj_weight = 100
        self.obj_priority = 1
        self.alpha_scale = 0.05


class AUAF:
    def __init__(self, supply, demand, edge):
        # load data
        self.params = Params()
        self.func = Func(self.params)
        self.alpha_j = None
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
        # beta_i
        beta_i = self.func.calculate_beta_i(mat_ctr, mat_gamma, self.alpha_j, self.theta_j)
        for i in range(mat_ctr.shape[0]):
            ctr_list, gamma_list = mat_ctr[i], mat_gamma[i]
            x_ij = self.func.calculate_x_ij(ctr_list, gamma_list, self.alpha_j, beta_i[i], self.theta_j)
            # price
            budget_remain = self.budget_max > 0.0
            pr = x_ij > 0
            price = x_ij
            bid_final = gamma_list * pr * price * budget_remain

            if np.sum(bid_final) == 0:
                continue

            cum_prob = np.cumsum(bid_final)
            #prob1 = random.random()
            #if prob1 >= cum_prob[-1]:
            #    continue
            cum_prob /= cum_prob[-1]
            prob2 = random.random()
            win_idx = self.func.search(prob2, cum_prob)

            self.cost_last[win_idx] += 1
            self.cost_accu[win_idx] += 1
            self.budget_max[win_idx] -= 1
            self.ctr_total[win_idx] += ctr_list[win_idx]

    def _update_dual(self):
        if self.params.T - self.t in (0, 1):
            self.alpha_j *= 0.95
            return
        # compute gradient
        eta = self.params.eta
        cost_last = self.cost_last

        g = (self.cost_expect - cost_last) * eta

        alpha_update = self.alpha_j - g

        alpha_update = np.clip(alpha_update, 0.001, 0.999)
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
        eta = round(self.params.eta, 3)
        finish_ratio = round(np.sum(self.cost_last) /
                             (np.sum(self.cost_expect) + 1e-6), 3)
        alpha_dual_avg = round(np.mean(self.alpha_j), 3)
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
        self.alpha_j, self.theta_j, self.cost_accu, self.cost_last, self.budget_left, \
        self.cost_expect, self.budget_max, self.ctr_total, self.smooth_loss = (
            np.ones(self.demand_num, dtype=np.float64) for _
            in range(9))

    def _init_before_start(self):
        self.cost_total_avg = self.budgets * 1.0 / self.params.T
        self.cost_accu *= 0
        self.cost_last *= 0
        self.ctr_total *= 0
        self.smooth_loss *= 0

        self.alpha_j = self.params.obj_weight + self.params.obj_lambda * self.ctr_org.max(
            axis=0).reshape(self.demand_num) * self.params.alpha_scale
        self.theta_j = self.budgets / (self.gamma.sum(axis=0).reshape(self.demand_num) + 0.0001)


if __name__ == '__main__':
    supply_file, demand_file = ['./data/supply_demand_data.txt'] * 2
    supply = Supply(supply_file)
    demand = Demand(demand_file)
    edge = Edge(supply_file, demand, supply)
    op = AUAF(supply, demand, edge)
    op.alloc_all(100)
