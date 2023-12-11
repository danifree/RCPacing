import math

import random
import numpy as np
from scipy import stats, special

from data.data_loader import *


class Func:
    def __init__(self, params):
        self.params = params

    def pct_estimate(self, data, momentum=None, lamb_before=None):
        """
        :param data:  np.narray
        :return:  bc_lamb, mean, std
        """

        data = data.reshape(-1, ) + 1e-6
        data_boxc, bc_lamb = stats.boxcox(data)
        if momentum is None or momentum <= 0.0:
            return bc_lamb, data_boxc.mean(), data_boxc.std() * (1 + self.params.epsilon)
        bc_lamb = bc_lamb * momentum + lamb_before * (1.0 - momentum)
        data_boxc = stats.boxcox(data, bc_lamb)
        return bc_lamb, data_boxc.mean(), data_boxc.std() * (1 + self.params.epsilon)

    def forward(self, x, bc_lamb, avg, std):
        x = (stats.boxcox(x, bc_lamb) - avg) / std
        return stats.norm.cdf(x)

    def ptr_pct(self, ptr_base, alpha_pct, ctr_pct):
        fp = self._fp(alpha_pct)
        fv = self._fv(alpha_pct, ctr_pct)
        return np.clip(ptr_base * fp * fv, 0.0001, 1.0)

    def ptr_fp(self, ptr_base, alpha_pct):
        fp = self._fp(alpha_pct)
        return np.clip(ptr_base * fp, 0., 1.0)

    def ptr_integ(self, ptr_base, alpha, n_sample=200):
        """
        :param ptr_base: scalar
        :param alpha: scalar
        :return: scalar
        """
        alpha = min(0.999, max(0.001, alpha))
        #         ctr = np.random.uniform(low=alpha, high=1.00, size=(n_sample,))
        ctr = np.linspace(alpha, 1.0, num=n_sample)
        ptr = self.ptr_pct(np.array([ptr_base] * n_sample), np.array([alpha] * n_sample), ctr)
        return np.mean(ptr) * (1.0 - alpha)

    def get_alpha_bisection(self, ptr_base, alpha, adjust_ratio):
        if adjust_ratio == 1.0:
            return alpha
        if alpha < 0.01 and adjust_ratio > 1.0:
            return 0.0
        elif alpha > 0.99 and adjust_ratio < 1.0:
            return alpha
        left = 0.001 if adjust_ratio > 1.0 else alpha
        right = 0.999 if adjust_ratio < 1.0 else alpha
        integ_target = self.ptr_integ(ptr_base, alpha) * adjust_ratio
        integ_left = self.ptr_integ(ptr_base, left)
        integ_right = self.ptr_integ(ptr_base, right)
        if integ_left < integ_target:
            return left
        if integ_right > integ_target:
            return right
        cnt = 0
        while right - left > 0.001 and cnt < 100 and integ_left > integ_right:
            cnt += 1
            left_ratio = min(0.99,
                             max(0.01, math.fabs((integ_right - integ_target) / (integ_left - integ_right))))
            mid = left * left_ratio + right * (1.0 - left_ratio)
            integ_mid = self.ptr_integ(ptr_base, mid)
            if integ_mid == integ_target:
                return integ_mid
            elif integ_mid < integ_target:
                right, integ_right = mid, integ_mid
            else:
                left, integ_left = mid, integ_mid
        return (left + right) / 2.0

    def _fv(self, alpha_pct, ctr_pct):
        return np.clip(ctr_pct - alpha_pct, 0.0, None) * self.params.k + self.params.k_base

    def _fp(self, alpha_pct):
        flag = alpha_pct <= self.params.p_ub
        speed_up = self.params.r_max ** ((self.params.p_ub - alpha_pct) / self.params.p_ub)
        slow_down = self.params.r_min ** ((self.params.p_ub - alpha_pct) / (self.params.p_ub - 1.0))
        return flag * speed_up + (1 - flag) * slow_down


class Params:
    def __init__(self):
        self.T = 50  # dual update total times

        self.p_ub = 0.9  # safe percentile upper bound
        # win-rate
        self.wr = 0.15
        self.eptr_init = 0.4
        self.r_sub = 1.75
        self.eptr_grad_fix = True

        # bid_price
        self.price_ctr = True

        # learning-rate
        self.eta = 0.1
        self.eta1 = 0.2
        self.eta_expo_decay = 0.85
        self.step_size_max = 0.05
        self.clip_dyn_range = 1.0
        self.clip_dyn_switch = True
        # box-cox
        self.epsilon = 0.1
        self.boxcox_update_switch = False
        self.boxcox_momentum = 0.1  # boxcox update momentum

        # fp & fv
        self.r_max = 50
        self.r_min = 0.25
        self.k = 10  # k(ctr-a) + k_base
        self.k_base = 0.0

        # max budget ratio
        self.linear_inc = True
        self.max_budget_ratio = [1.2, 4.0]

        # update dual
        self.dist_func = 'itakura'  # 'euclidean' or 'itakura'
        self.dist_a = 1.5


class RCPacing:
    def __init__(self, supply, demand, edge):
        # load data
        self.func = None
        self.params = None
        self.alpha_pct = None
        self.t = 0
        self.supply = supply
        self.demand = demand
        self.edge = edge
        self.demand_num = self.demand.demand_num
        self.supply_num = self.supply.supply_num
        self.budgets = self.demand.demand_mat.reshape(self.demand_num)

    def alloc_all(self, n=None):
        self.params = Params()
        self.func = Func(self.params)
        self._init_df()
        self._init_before_start()
        batch_size = self.supply_num // self.params.T
        for i in range(self.params.T):
            self.t += 1
            logs_ctr = self.edge.ctr[max(0, i - 1) * batch_size:i * batch_size]
            logs_gamma = self.edge.gamma[max(0, i - 1) * batch_size:i * batch_size]
            self._update_boxcox(logs_ctr, logs_gamma)
            supply_ctr = self.edge.ctr[i * batch_size: (i + 1) * batch_size]
            supply_gamma = self.edge.gamma[i *
                                           batch_size: (i + 1) * batch_size]
            self._update_budget()
            self._pct2dual()
            self._alloc_t(supply_ctr, supply_gamma)
            self._metric()
            self._update_dual_pct()
            self._update_eptr()
            if n is not None and i >= n:
                break

    def _alloc_t(self, mat_ctr, mat_gamma):
        for i in range(mat_ctr.shape[0]):
            ctr_list, gamma_list = mat_ctr[i], mat_gamma[i]
            # calculate ptr_fv
            ctr_pct = self.func.forward(ctr_list, self.boxcox_lambda, self.boxcox_avg,
                                        self.boxcox_std)
            ptr_pct_t = self.func.ptr_pct(self.ptr_base, self.alpha_pct, ctr_pct) * self.ptr_eptr
            # close pacing in the last periods
            #if self.params.T - self.t <= 1:
            #    ptr_pct_t = np.clip(ptr_pct_t * 2 ** (self.t - self.params.T + 5), 0., 1.)
            ptr_res = np.random.rand(self.demand_num) <= ptr_pct_t
            # price
            budget_remain = self.budget_max > 0.0
            bid = (ctr_list - self.alpha_dual)
            pr = bid > 0.0
            price = (ctr_list - self.alpha_dual)
            if self.params.price_ctr:
                price = ctr_list
            bid_final = gamma_list * ptr_res * pr * price * budget_remain
            idx_max = np.argmax(bid_final)
            if bid_final[idx_max] > 0:
                self.cost_last[idx_max] += 1
                self.cost_accu[idx_max] += 1
                self.budget_max[idx_max] -= 1
                self.ctr_total[idx_max] += ctr_list[idx_max]

    def _update_dual_pct(self):
        if self.params.T - self.t in (0, 1):
            self.alpha_dual *= 0.95
            return
        if self.t % 5 == 0:
            self.params.eta = self.params.eta * self.params.eta_expo_decay
        # compute gradient
        eta = self.params.eta1 if self.t == 1 else self.params.eta
        cost_last = self.cost_last
        # eptr_grad_fix should exclude the effect of eptr init
        if self.t > 3 and self.params.eptr_grad_fix:
            cost_last = self.cost_last / self.ptr_eptr
        g = (self.cost_expect - cost_last) / \
            (self.cost_expect + 0.1) * eta
        if self.params.dist_func == 'itakura':
            adapter = np.clip(g * (self.params.dist_a - self.alpha_pct), None, 0.999)
            g *= (self.params.dist_a - self.alpha_pct) ** 2 / (1 - adapter)

        # static clip
        g = np.clip(g, -self.params.step_size_max, self.params.step_size_max)

        # dynamic clip
        if self.params.clip_dyn_switch:
            enlarge_ratio = np.clip(
                self.cost_expect, 0.1, None) / np.clip(self.cost_last, 0.1, None)
            clip_range = np.array(
                [self.params.clip_dyn_range] * self.demand_num)
            adjust_list = np.clip(
                1.0 + clip_range * (enlarge_ratio - 1.0), 0.1, None)
            for j in range(self.demand_num):
                adjust_list[j] = self.func.get_alpha_bisection(
                    self.ptr_base[j], self.alpha_pct[j], adjust_list[j])
            alpha_res = np.clip(
                np.abs(adjust_list - self.alpha_pct), 0.002, None)
            alpha_update = np.maximum.reduce(
                [np.minimum.reduce([self.alpha_pct - g, self.alpha_pct + alpha_res]), self.alpha_pct - alpha_res])
        else:
            alpha_update = self.alpha_pct - g

        alpha_update = np.clip(alpha_update, 0.001, 0.999)
        self.alpha_diff = np.abs(self.alpha_pct - alpha_update)
        self.alpha_pct = alpha_update

    def _update_boxcox(self, data, gamma):
        if self.t == 1 or not self.params.boxcox_update_switch:
            return
        valid_idx = gamma == 1
        for j in range(self.demand_num):
            data_j = data[:, j]
            data_idx = valid_idx[:, j]
            self.boxcox_lambda[j], self.boxcox_avg[j], self.boxcox_std[j] = self.func.pct_estimate(
                data_j[data_idx], self.params.boxcox_momentum, self.boxcox_lambda[j])

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
        self.ptr_fp_stats = self.func.ptr_fp(self.ptr_base, self.alpha_pct)
        smooth_loss_mean = round(np.mean((self.smooth_loss / self.t) ** 0.5), 3)
        eta = round(self.params.eta, 3)
        alpha_diff = round(np.mean(self.alpha_diff), 4)
        finish_ratio = round(np.sum(self.cost_last) /
                             np.sum(self.cost_expect), 3)
        alpha_pct_avg = round(np.mean(self.alpha_pct), 3)
        alpha_dual_avg = round(np.mean(self.alpha_dual), 3)
        finish_total = round(np.sum(self.cost_accu) / np.sum(self.budgets), 5)
        ctr_all = round(np.sum(self.ctr_total), 0)
        ctr_avg = round(np.sum(self.ctr_total) /
                        (np.sum(self.cost_accu) + 1), 5)
        eptr = round(np.mean(self.ptr_eptr), 3)
        ptr_fp_stats = round(np.mean(self.ptr_fp_stats), 3)
        cost_last = round(np.sum(self.cost_last), 1)

        loss_this_round = np.round(
            np.mean((self.cost_last - self.cost_total_avg) ** 2), 1)
        print(f"round {self.t}:")
        print(
            f"finish ratio {finish_ratio},  eta {eta}, eptr {eptr}, finish total {finish_total}, "
            f"alpha_pct_avg {alpha_pct_avg}, alpha_dual_avg {alpha_dual_avg}, ctr_avg {ctr_avg}, ctr_all {ctr_all}, "
            f"smooth_loss {smooth_loss_mean},loss_this {loss_this_round}, ptr_fp_stats {ptr_fp_stats}, alpha_diff {alpha_diff}, cost_last {cost_last}")

    def _init_df(self):
        self.gamma = self.edge.gamma
        self.ctr_org = self.edge.ctr
        self.alpha_dual, self.alpha_pct, self.boxcox_avg, self.boxcox_std, self.boxcox_lambda, self.ptr_final, self.ptr_base, \
        self.ptr_eptr, self.cost_accu, self.cost_last, self.budget_left, self.cost_expect, self.budget_max, self.ctr_total, self.smooth_loss, self.alpha_diff = (
            np.ones(self.demand_num, dtype=np.float64) for _
            in range(16))

    def _init_before_start(self):
        self.cost_total_avg = self.budgets * 1.0 / self.params.T
        self.cost_accu *= 0
        self.cost_last *= 0
        self.ctr_total *= 0
        self.smooth_loss *= 0
        self.ptr_eptr *= self.params.eptr_init

        # init alpha_pct and ptr_base
        theta = self.budgets / \
                (self.gamma.sum(axis=0).reshape(self.demand_num) + 0.0001)
        ptr_expect = theta / (1.0 - self.params.p_ub)
        for i in range(self.demand_num):
            self.ptr_base[i] = min(1., ptr_expect[i] / self.params.wr)
            alpha_pct_init_base = ptr_expect[i] / self.params.wr
            self.alpha_pct[i] = self.params.p_ub if alpha_pct_init_base <= 1 else max(0, 1 - (
                    1 - self.params.p_ub) * alpha_pct_init_base)
        # init box-cox global
        stats_init = min(5 * 10000, self.ctr_org.shape[0])
        ctr = self.ctr_org[0:stats_init]
        gamma = self.gamma[0:stats_init]
        bc_lamb, avg, std = self.func.pct_estimate(ctr[gamma == 1])
        self.boxcox_lambda *= bc_lamb
        self.boxcox_avg *= avg
        self.boxcox_std *= std
        print("boxcox_glb:", round(bc_lamb, 3))

    def _pct2dual(self):
        alpha_t = stats.norm.ppf(self.alpha_pct) * \
                  self.boxcox_std + self.boxcox_avg
        self.alpha_dual = special.inv_boxcox(alpha_t, self.boxcox_lambda)

    def _update_eptr(self):
        if self.params.T - self.t in (0, 1):
            self.ptr_eptr = 1.0
        else:
            speed = (self.cost_last + 0.1) / (self.cost_expect + 0.1)
            self.ptr_eptr = np.clip(self.ptr_eptr * np.clip(self.params.r_sub / speed, None, self.params.r_sub), 0.01,
                                    1.0)


if __name__ == '__main__':
    supply_file, demand_file = ['./data/supply_demand_data.txt'] * 2
    supply = Supply(supply_file)
    demand = Demand(demand_file)
    edge = Edge(supply_file, demand, supply)
    op = RCPacing(supply, demand, edge)
    op.alloc_all(100)
