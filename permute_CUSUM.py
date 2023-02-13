import numpy as np
from scipy.stats import norm

def sim_one_rl(p, H, k):
    t = 0
    S = np.zeros(p)
    while max(S) < H:
        t = t + 1
        r = np.random.permutation(p)
        z = norm.ppf((r + 1 - np.random.rand(p)) / p)
        S = np.array([max(u-k,0) for u in (S - z)])
    return t

def sim_N_rl(p,H,k,N):
    rls = np.zeros(N)
    for j in range(N):
        rls[j] = sim_one_rl(p,H,k)
    arl = np.mean(rls)
    sd = np.std(rls) / np.sqrt(N)
    return arl, sd, rls

def sim_rl_seq(p,H,k,N,arl0):
    rls = []
    n = 0
    while True:
        n = n + 1
        rls.append(sim_one_rl(p,H,k))
        arl = np.mean(rls)
        sd = np.std(rls) / np.sqrt(n)
        if n > 30 and (arl-arl0) / sd > 3:
            return -1, arl, sd, rls # drop the control limit
        if n > 30 and (arl0 - arl) / sd > 3:
            return +1, arl, sd, rls  # increase the control limit
        if n > N:
            return 0, arl, sd, rls


def findbound(p,k,N, arl0):
    H = 5
    prevH = H
    action, arl, sd, rls = sim_rl_seq(p,H,k,N,arl0)
    print(f"-> Find Bounds: H = {H:.2f}, arl = {arl:.2f}, arl0 = {arl0}")
    prevarl = arl
    if action == -1:
        while True:
            prevH = H
            prevarl = arl
            H = H / 1.05
            action, arl, sd, rls = sim_rl_seq(p,H,k,N,arl0)
            print(f"-> Find Bounds: H = {H:.2f}, arl = {arl:.2f}, arl0 = {arl0}")
            if action == 0:
                return None, arl, sd, rls
            if action == 1:
                break
    elif action == 1:
        while True:
            prevH = H
            prevarl = arl
            H = H * 1.05
            action, arl, sd, rls = sim_rl_seq(p,H,k,N,arl0)
            print(f"-> Find Bounds: H = {H:.2f}, arl = {arl:.2f}, arl0 = {arl0}")
            if action == 0:
                return None, arl, sd, rls
            if action == -1:
                break
    return prevH, prevarl, H, arl

def calH (p,k,N, arl0):
    H1, arl1, H2, arl2 = findbound(p,k,N,arl0)
    if H1 == None:
        return arl1, H2, arl2 # I abused the function: when action is 0, the first returned element is Null and the next three are arl, sd, rls. ARLs is found with early stopping.
    if arl1 < arl2:
        Hl = H1; arll = arl1
        Hh = H2; arlh = arl2
    else:
        Hh = H1; arlh = arl1
        Hl = H2; arll = arl2

    while True:
        r = (arlh - arl0) / (arl0 - arll)
        Htest = Hl + 1 / (1+r) * (Hh - Hl)
        action, arl, sd, rls = sim_rl_seq(p,H,k,N,arl0)
        print(f'Bounds: {Hh:.2f} ({arlh:.2f}),{Hl:.2f} ({arll:.2f}). Tested {Htest:.2f} ({arl:.2f}; {sd:.2f})')
        if action == 0:
            return arl, sd, rls, Htest
        else:
            if arl > arl0:
                arlh = arl
                Hh = Htest
            else:
                arll = arl
                Hl = Htest


class Monitor: # this is used for reading the residual ranks in real time.
    def __init__(self, p, H, k):
        self.H = H
        self.k = k
        self.p = p
        self.S = np.zeros(p) # detect time
        self.t = 0 # detect time
        self.alarm = False
    def reset(self):
        self.alarm = False
        self.t = 0
        self.S = np.zeros(self.p)
    def newobs(self, r):
        self.t = self.t + 1
        np.array(r) + 1 - np.random.rand(self.p)
        z = norm.ppf((np.array(r) + 1 - np.random.rand(self.p)) / self.p)
        self.S = np.array([max(u-self.k,0) for u in (self.S - z)])
        if max(self.S) > self.H:
            self.alarm = True
            return np.argmax(self.S), self.t
        else:
            return None, self.t
    def status(self):
        print(f'p={self.p}; H={self.H}; k={self.k}; t={self.t}')
        print(self.S)


arl,sd, rls = calH(10,.5,5000,50)

cumon = Monitor(3,2,.3)
cumon.reset()
cumon.newobs([2,1,0])
cumon.status()
cumon.newobs([2,1,0])
cumon.status()
