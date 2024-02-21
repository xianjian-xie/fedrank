import numpy as np
from scipy.stats import norm

# p: num of clients, H: upper bound, k: constant coefficient
def sim_one_rl(p, H, k):
    MAXRL = 1000
    t = 0
    S = np.zeros(p)
    while max(S) < H:
        t = t + 1
        r = np.random.permutation(p)
        z = norm.ppf((r + 1 - np.random.rand(p)) / p)
        S = np.array([max(u-k,0) for u in (S - z)])
        if t > MAXRL:
            return -1
    return t

def sim_N_rl(p,H,k,N):
    rls = np.zeros(N)
    for j in range(N):
        rls[j] = sim_one_rl(p,H,k)
        if rls[j] == -1:
            return -1, []
    arl = np.mean(rls)
    sd = np.std(rls) / np.sqrt(N)
    return 0, [arl, sd, rls]

def sim_rl_seq(p,H,k,N,arl0):
    rls = []
    n = 0
    while True:
        n = n + 1
        rl = sim_one_rl(p,H,k)
        if rl == -1:
            return -1, []
        rls.append(rl)
        arl = np.mean(rls)
        sd = np.std(rls) / np.sqrt(n)
        if n > 30 and (arl-arl0) / sd > 3:
            return -1, [arl, sd, rls] # decrease the control limit
        if n > 30 and (arl0 - arl) / sd > 3:
            return +1, [arl, sd, rls]  # increase the control limit
        if n > N:
            return 0, [arl, sd, rls]

# find bound of H(control limit)
def findbound(p,k,N, arl0):
    def actiontxt (action):
        txt = "Drop H" if action == -1 else "Increase H" if action == 1 else "Optimal H found"
        return txt

    H = 5
    prevH = H
    print(f"Try bounds: trial H = {H:.2f}.")
    action, results = sim_rl_seq(p,H,k,N,arl0)
    if not results:
        print(f"-> Bounds not found. ARL exceeds MAXRL.  {actiontxt(action)}.")
    else:
        print(f"-> Bounds found: H = {H:.2f}, arl = {results[0]:.2f} ({results[1]:.2f}), arl0 = {arl0}. {actiontxt(action)}. ")

    if action == 0:
        return action, results

    prevresults = results

    if action == -1:
        while True:
            prevH = H
            prevresults = results
            H = H * .9
            print(f"Try bounds: trial H = {H:.2f}.")
            action, results = sim_rl_seq(p, H, k, N, arl0)
            if not results:
                print(f"-> Bounds not found. ARL exceeds MAXRL.  {actiontxt(action)}.")
            else:
                print(
                    f"-> Bounds found: H = {H:.2f}, arl = {results[0]:.2f} ({results[1]:.2f}), arl0 = {arl0}. {actiontxt(action)}. ")
            if action == 0:
                return 0, results
            if action == 1:
                break

    elif action == 1:
        while True:
            prevH = H
            prevresults = results
            H = H * 1.1
            print(f"Try bounds: trial H = {H:.2f}.")
            action, results = sim_rl_seq(p, H, k, N, arl0)
            if not results:
                print(f"-> Bounds not found. ARL exceeds MAXRL.  {actiontxt(action)}.")
            else:
                print(
                    f"-> Bounds found: H = {H:.2f}, arl = {results[0]:.2f} ({results[1]:.2f}), arl0 = {arl0}. {actiontxt(action)}. ")
            if action == 0:
                return 0, results # optimal H is found
            if action == -1:
                break
    return 1, [prevH, prevresults, H, results] # The bounds of H is found.

def calH (p,k,N, arl0):
    status, results = findbound(p,k,N,arl0)
    if status == 0:
        return results

    [H1, [arl1, _, _], H2, [arl2, _, _]] = results

    if arl1 < arl2:
        Hl = H1; arll = arl1
        Hh = H2; arlh = arl2
    else:
        Hh = H1; arlh = arl1
        Hl = H2; arll = arl2

    while True:
        r = (arlh - arl0) / (arl0 - arll)
        Htest = Hl + 1 / (1+r) * (Hh - Hl)
        print(f'Search H: trial H = {Htest:.2f}')
        action, results = sim_rl_seq(p,Htest,k,N,arl0)
        if action == 0:
            print(
                f"-> H found: H = {Htest:.2f}, arl = {results[0]:.2f} ({results[1]:.2f}), arl0 = {arl0}. ")
            return results, Htest

        else:
            if results[0] > arl0:
                arlh = results[0]
                Hh = Htest
                print(
                    f"-> Update search bound: H_low = {Hl:.2f}, arl_low = {arll:.2f}; H_hi = {Hh:.2f}, arl_hi = {arlh:.2f} ")
            else:
                arll = results[0]
                Hl = Htest


class Monitor: # this is used for reading the residual ranks in real time.
    def __init__(self, p, H, k):
        self.H = H  # bound, threshold
        self.k = k  # parameter to control cusum curve not to flow up
        self.p = p  # number of client
        self.S = np.zeros(p)
        self.l = list(range(0,p))
        self.t = 0 # detect time
        self.alarm = False
    def reset(self):
        self.alarm = False
        self.t = 0
        self.S = np.zeros(self.p)
        self.l = list(range(0,self.p))
    def newobs(self, r):
        self.t = self.t + 1
        np.array(r) + 1 - np.random.rand(self.p)
        z = norm.ppf((np.array(r) + 1 - np.random.rand(self.p)) / self.p)
        # print('z is',z)
        self.S = np.array([max(u-self.k,0) for u in (self.S - z)])
        # print('s shape', self.S.shape, type(self.S))
        for i in range(len(self.l)):
            # print('h1')
            if self.S[self.l[i]] > self.H and self.t>3:
                # print('h2')
                self.alarm = True
                return self.l.pop(i), self.t
        # print('h3')
        return None, self.t
    def status(self):
        print(f'p={self.p}; H={self.H}; k={self.k}; t={self.t}; alarm={self.alarm}')
        print(self.S)


# results = calH(20,.5,1000,50)

# cumon = Monitor(3,3,.3)
# cumon.reset()
# cumon.newobs([2,1,0])
# cumon.status()
# cumon.newobs([2,1,0])
# cumon.status()

# p,  k:0.4,0.5,0.6,     N:1000,     arl0:30,50

# results = calH(5,0.6,1000,50)
# print('results is', results)


# cumon = Monitor(5,3.84,.4)
# cumon.reset()
# for i in range(15):
#     output = cumon.newobs([1,0,2,3,4])
#     cumon.status()
#     if output[0]==1:
#         print('output is', output, type(output[0]), type(output[1]))
#         cumon.reset()
