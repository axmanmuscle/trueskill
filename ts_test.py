import numpy as np
import ts_graph as graph
import trueskill as ts

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

    def __init__(self, mu = None, sigma = None):
        if mu is None:
            self.mu = GLOBAL_MU
        else:
            self.mu = mu

        if sigma is None:
            self.sigma = GLOBAL_SIGMA
        else:
            self.sigma = sigma

        self.mean = self.mu
        self.sd = self.sigma

    def __repr__(self):
        s = "Rating class with mean {} and sd {}".format(self.mean, self.sd)
        return(s)

def make_players_test():
    p1 = Rating(mu=110, sigma=10)
    p2 = Rating(mu=100, sigma=20)
    p3 = Rating(mu=95, sigma=30)

    p4 = Rating(mu=130, sigma=5)
    p5 = Rating(mu=120, sigma=20)
    p6 = Rating(mu=105, sigma=25)

    players = [p1, p2, p3, p4, p5, p6]

    t1 = [p1, p2, p3]
    t2 = [p4, p5, p6]
    teams = [t1, t2]
    return players, teams

def make_players_test2():
    p1 = Rating(mu=20, sigma=5)

    p2 = Rating(mu=30, sigma=10)


    players = [p1, p2]

    t1 = [p1]
    t2 = [p2]
    teams = [t1, t2]
    return players, teams

n_players = 6
n_teams = 2

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
        start = team_sizes[team - 1]
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

min_delta = 1e-6
delta = 10

team_diff_len = len(team_diff_layer)

i = 1
while delta > min_delta:
    print('schedule iter {}'.format(i))
    print('delta: {}'.format(delta))
    if team_diff_len == 1:
        team_diff_layer[0].down()
        delta = trunc_layer[0].up()
    ## TODO: add for more teams
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