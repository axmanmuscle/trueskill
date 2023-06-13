import numpy as np
import ts_graph as graph
from ts_run import rate

"""
we need to 
    - form players
    - form all levels of the factor graph
    - pass messages
    - read skill out

eventually we'll want to be able to pull players and teams from a file?
"""

"""
TODO: continue forming the factor graph
maybe functionize it all so we can reproduce faster
"""


GLOBAL_MU = 100
GLOBAL_SIGMA = 20

class Rating():
    """
    class for individual player ratings
    """

    def __init__(self, mu = None, sigma = None, name = '', teamid = -1):
        if mu is None:
            print('setting default mu')
            self.mu = GLOBAL_MU
        else:
            self.mu = mu

        if sigma is None:
            print('setting default sigma')
            self.sigma = GLOBAL_SIGMA
        else:
            self.sigma = sigma

        self.mean = self.mu
        self.sd = self.sigma
        self.name = name
        self.teamid = teamid

    def __repr__(self):
        s = "Player {} on team {} with mean {} and sd {}".format(self.name, self.teamid, self.mean, self.sd)
        return(s)
    
    def set(self, variable):
        """
        update values at end of TS run
        """
        self.mean = variable.mu
        self.sd = variable.sigma
        print('successfully set with new mean {} and sigma {}'.format(self.mean, self.sd))

def make_players_test():
    p1 = Rating(mu=110, sigma=10)
    p2 = Rating(mu=100, sigma=20)
    p3 = Rating(mu=95, sigma=30)

    p4 = Rating(mu=130, sigma=5)
    p5 = Rating(mu=120, sigma=20)
    p6 = Rating(mu=105, sigma=25)

    p7 = Rating(mu=115, sigma=15)
    p8 = Rating(mu=108, sigma=20)
    p9 = Rating(mu=110, sigma=25)

    players = [p1, p2, p3, p4, p5, p6, p7, p8, p9]

    t1 = [p1, p2, p3]
    t2 = [p4, p5, p6]
    t3 = [p7, p8, p9]
    teams = [t1, t2, t3]
    return players, teams

def make_players_test2():
    p1 = Rating(mu=20, sigma=5)

    p2 = Rating(mu=30, sigma=10)

    players = [p1, p2]

    t1 = [p1]
    t2 = [p2]
    teams = [t1, t2]
    return players, teams

n_players = 9
n_teams = 3

mu = 100
sigma = 20
beta = 10
tau = 2

players, teams = make_players_test()

rating_vars = [graph.Variable() for i in range(n_players)]
perf_vars = [graph.Variable() for i in range(n_players)]
team_perf_vars = [graph.Variable() for i in range(n_teams)]
team_diff_vars = [graph.Variable() for i in range(n_teams - 1)]

team_sizes = [len(i) for i in teams]
sum_team_sizes = np.cumsum(team_sizes)

prior_layer = []
for rating_var, player in zip(rating_vars, players):
    var = graph.PriorFactor(rating_var, player, tau)
    prior_layer.append(var)

perf_layer = []
for rating_var, perf_var in zip(rating_vars, perf_vars):
    var = graph.LikelihoodFactor(rating_var, perf_var, beta**2)
    perf_layer.append(var)

team_perf_layer = []
for team, team_perf_var in enumerate(team_perf_vars):
    if team > 0:
        start = sum_team_sizes[team - 1]
    else:
        start = 0
    team_size = team_sizes[team]
    child_perf_vars = perf_vars[start:start+team_size]
    coeffs = [1 for i in range(team_size)]
    var = graph.SumFactor(team_perf_var, child_perf_vars, coeffs)
    team_perf_layer.append(var)

team_diff_layer = []
for team, team_diff_var in enumerate(team_diff_vars):
    var = graph.SumFactor(team_diff_var, team_perf_vars[team:team+2],
                           [1, -1])
    team_diff_layer.append(var)
    
trunc_layer = []
"""
for now, we're not going to allow ties (thinking of Apex)
We may want to allow ties e.g. for teams 12-20 (ALGS style?) or 
similar to balance scores better
"""
for x, team_diff_var in enumerate(team_diff_vars):
    draw_prob = 0
    draw_margin = 1e-16
    v_func, w_func = graph.v_win, graph.w_win
    var = graph.TruncateFactor(team_diff_var, v_func=v_func,
                                w_func=w_func, draw_margin=draw_margin)
    trunc_layer.append(var)

"""
With variables and layers all built, time to run the schedule
"""

for i,f in enumerate(prior_layer):
    f.down()
for i,f in enumerate(perf_layer):
    f.down()
for i,f in enumerate(team_perf_layer):
    f.down()

min_delta = 0.001
delta = 10

team_diff_len = len(team_diff_layer)

"""
iterate the non-gaussians until they settle down
"""
i = 1
while delta > min_delta: ##may want max iters here
    if team_diff_len == 1:
        team_diff_layer[0].down()
        delta = trunc_layer[0].up()
    else:
        delta = 0
        for x in range(team_diff_len - 1):
            team_diff_layer[x].down()
            delta = max(delta, trunc_layer[x].up())
            team_diff_layer[x].up(1)  # up to right variable
        for x in range(team_diff_len - 1, 0, -1):
            team_diff_layer[x].down()
            delta = max(delta, trunc_layer[x].up())
            team_diff_layer[x].up(0)  # up to left variable
    if delta <= min_delta:
        break
    i += 1

# up both ends
team_diff_layer[0].up(0)
team_diff_layer[team_diff_len - 1].up(1)

# up the remainder
for f in team_perf_layer:
    for x in range(len(f.inputs)):
        f.up(x)
for f in perf_layer:
    f.up()
    
print([r for r in rating_vars])

r1 = rate(teams, [2, 3, 1])

print(r1)