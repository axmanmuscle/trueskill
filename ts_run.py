import numpy as np
import ts_graph as graph

beta = 1000/6 # sigma / 2
tau = 1000/300 # sigma / 100

    
def rate(unsorted_teams, rankings):
    """
    this function runs the rating/schedule
    unsorted_teams should be a list of lists, where each sublist is a collection of Rating objects (representing the players on each team)
    rankings is a list of numbers ranking the teams at the end of the game
        1 > 2 > 3 .... > n
        where 1 is the winner

    there's some bookkeeping involved in taking in arbitrarily sorted teams and keeping track of them
    a couple options:
        - some funky sorting logic and then sorting back to the order they were handed in
        - giving each team a "name" so user knows which team is which
    or we could just required that the teams come in in the correct order and we don't have to deal with it?

    but i think we might want to add team/player names 
    """

    sorted_teams = [s for s in zip(unsorted_teams, rankings)]
    sorted_teams = sorted(sorted_teams, key=lambda x: x[1])
    teams = [t[0] for t in sorted_teams]

    players = [p for team in teams for p in team]
    n_players = sum([len(i) for i in teams])
    n_teams = len(teams)

    ## create variables

    rating_vars = [graph.Variable() for i in range(n_players)]
    perf_vars = [graph.Variable() for i in range(n_players)]
    team_perf_vars = [graph.Variable() for i in range(n_teams)]
    team_diff_vars = [graph.Variable() for i in range(n_teams - 1)]

    team_sizes = [len(i) for i in teams]
    sum_team_sizes = np.cumsum(team_sizes)

    ## build layers of graph
    # prior layer
    prior_layer = []
    for rating_var, player in zip(rating_vars, players):
        var = graph.PriorFactor(rating_var, player, tau)
        prior_layer.append(var)

    # performance layer
    perf_layer = []
    for rating_var, perf_var in zip(rating_vars, perf_vars):
        var = graph.LikelihoodFactor(rating_var, perf_var, beta**2)
        perf_layer.append(var)

    # team performance layer
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

    # team difference layer
    team_diff_layer = []
    for team, team_diff_var in enumerate(team_diff_vars):
        var = graph.SumFactor(team_diff_var, team_perf_vars[team:team+2],
                            [1, -1])
        team_diff_layer.append(var)
        
    # truncation layer
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

    min_delta = 0.0001
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
    for f in prior_layer:
        f.up()

    # figure out how to group this up so it comes back out as teams like the input
    return teams 