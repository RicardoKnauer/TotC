from mesa import Agent
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.time import RandomActivation, StagedActivation

import numpy as np
import networkx as nx
import random
import scipy.stats as stats


GRID_SIZE = 33          # 33x33 grid
GROWTH_RATE = .1        # grass regrowth rate
MAX_STEPS = 5           # max number of steps per sheep and time step
P = 10                  # sheep price
REQU = 1                # amount of grass a sheep needs to eat per time step
VEG_MAX = GRID_SIZE**2  # maximum amount of vegetation


def g(grass):
    """
    Returns the new amount of grass at a patch after a timestep
    """
    return grass + GROWTH_RATE * grass * (VEG_MAX - grass) / VEG_MAX


class Walker(Agent):
    """
    Agent that performs a guided walk
    """

    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)

        self.pos = pos
        self.unique_id = unique_id

    def move(self):
        """
        Move to the grass agent that has the highest density in the area
        """
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        # Select grass path with highest density
        highest_density = 0
        best_choice = None
        for step in possible_steps:
            this_cell = self.model.grid.get_cell_list_contents([step])
            for agent in this_cell:
                if isinstance(agent, Grass):
                    if agent.density > highest_density:
                        highest_density = agent.density
                        best_choice = step
        # Move agent to grass path with highest density
        self.model.grid.move_agent(self, best_choice)


class Sheep(Walker):
    """
    The sheep agent performs the guided walk as defined in Walker and eats grass
    """
    sheepdeaths = 0

    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model, pos)
        self.saturation = 1
        self.owner = None

    def die(self):
        self.model.remove_agent(self)
        self.owner.stock.remove(self)
        Sheep.sheepdeaths += 1

    def step(self):
        """
        Lower the sheep's saturation by .5 and eat grass until the saturation
        level is back to at least REQU. It will eat grass with a maximum of
        MAX_STEPS times.
        If the saturation is below REQU it will die.
        """
        i = 0
        self.saturation -= .5
        # Sheep wander MAX_STEPS number of steps around to fulfill their food requirement
        while i < MAX_STEPS and self.saturation < REQU:
            this_cell = self.model.grid.get_cell_list_contents([self.pos])
            for agent in this_cell:
                if isinstance(agent, Grass):
                    self.saturation += (agent.density - .05)
                    agent.fade()
            self.move()
            i += 1
        if self.saturation < REQU:
            self.die()


class Grass(Agent):

    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.pos = pos
        self.density = random.random()

    def fade(self):
        """
        Grass fades.
        """
        tmp = self.density
        self.density = 0.05
        return tmp

    def next_density(self):
        """
        Calculate the density that needs to be added to the current density
        to get a logistic growth.
        """
        return GROWTH_RATE * self.density * (1 - self.density)

    def step(self):
        """
        Grass regrows logistically.
        """
        self.density += self.next_density()


class Herdsman(Agent):
    i = 0
    x = None

    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.pos = pos
        self.unique_id = unique_id
        self.index = Herdsman.i

        # Decision at each time step
        self.a = []
        # Weight for social network
        self.friendship_weights = []
        # Number of cattle owned at each time step
        self.k = []
        self.stock = []

        # Truncated normal distribution of psychosocial factors
        self.l_coop = stats.truncnorm((0 - self.model.l_coop) / .1, (1 - self.model.l_coop) / .1,
                                      self.model.l_coop, scale=.1).rvs()
        self.l_fairself = stats.truncnorm((0 - self.model.l_fairself) / .1, (1 - self.model.l_fairself) / .1,
                                          self.model.l_fairself, scale=.1).rvs()
        self.l_fairother = stats.truncnorm((0 - self.model.l_fairother) / .1, (1 - self.model.l_fairother) / .1,
                                           self.model.l_fairother, scale=.1).rvs()
        self.l_posrecip = stats.truncnorm((0 - self.model.l_posrecip) / .1, (1 - self.model.l_posrecip) / .1,
                                          self.model.l_posrecip, scale=.1).rvs()
        self.l_negrecip = stats.truncnorm((0 - self.model.l_negrecip) / .1, (1 - self.model.l_negrecip) / .1,
                                          self.model.l_negrecip, scale=.1).rvs()
        self.l_conf = stats.truncnorm((0 - self.model.l_conf) / .1, (1 - self.model.l_conf) / .1,
                                      self.model.l_conf, scale=.1).rvs()

        # The next herdsman will get a higher index.
        Herdsman.i += 1

        for i in range(model.initial_sheep_per_herdsmen):
            self.add_sheep()

    def add_sheep(self):
        """Spawn a new sheep at a random location and assign it to this herdsman"""
        x = random.randrange(self.model.width)
        y = random.randrange(self.model.height)
        sheep = self.model.new_agent(Sheep, (x, y))
        self.stock.append(sheep)
        sheep.owner = self

    def advance(self):
        """
        Think of what decision to make.
        This function gets executed before step()
        """
        self.decision = self.decide()

    def remove_sheep(self):
        """Remove one sheep from this herdsman."""
        if len(self.stock) != 0:
            sheep = self.stock.pop()
            self.model.remove_agent(sheep)

    def s(self, d, h, t):
        """
        Checks if the decision made at timestep `t' is equal to `d'.
        Same as the `s' function defined in Schindler.
        
        d : { -1, 0, 1 }
        h: herdsman
        t: time step
        """
        return 1 if len(h.a) > 0 and h.a[t] == d else 0

    def step(self):
        """
        Perform the decision made using advance()
        """
        if self.decision > 0:
            self.add_sheep()
        elif self.decision < 0:
            self.remove_sheep()

        self.a.append(int(self.decision))
        Herdsman.x[:] = 0


    def p_basic(self, x):
        """
        The basic payoff for a decision list. `x' will contain the decision that
        every herdsman makes. p basic will always return higher values if this
        herdsman is adding sheep or other herdsmen are removing sheep.
        It will return a lower value when other people add sheep.
        """
        N = self.model.get_herdsman_count()
        grass = self.model.get_grass_count()
        sheep = self.model.get_sheep_count()
        cost = (g(max(0, grass - sheep * REQU)) - g(max(0, grass - (sheep + sum(x)) * REQU))) * P / REQU
        add_cost_grass = sum(x) * P * (1 - grass / 1089)
        add_cost_death = sum(x) * P * (1 - self.model.sheep_survival_rate[-1])
        cost = cost + add_cost_death + add_cost_grass

        return (len(self.stock) + x[self.index]) * P - (cost / N)

    def p_coop(self, x):
        """
        The payoff when cooperation is taken into account. When `l_coop' is
        equal to 0 this function returns `p_basic'. When `l_coop' is 1 it will
        return the average `p_basic' of all other players for this decision list
        `x'. `l_coop' can be a float between 0 and 1. The social network weights
        are also taken into account.
        """
        sumA = 0
        count = 0
        for herdsman in self.model.herdsmen:
            if herdsman is not self:
                sumA += self.friendship_weights[count] * herdsman.p_basic(x)
                count += 1
        result = (1 - self.l_coop) * self.p_basic(x) + sumA * self.l_coop / sum(self.friendship_weights)

        return result

    def add_fair(self, x):
        """
        This function will return a higher value if `l_fairself' is high and 
        your decision will make you more equal to others, if others have more 
        sheep than you. If `l_fairother' is high this function will return a
        high value if you remove sheep and others have fewer sheep than you. The
        social network weights are also taken into account.
        """
        sumA, sumB, sumC = 0, 0, 0
        count = 0
        for herdsman in self.model.herdsmen:
            if herdsman is not self:
                sumA += max(herdsman.p_basic(x) - self.p_basic(x), 0)
                sumB += self.friendship_weights[count] * max(0, self.p_basic(x) - herdsman.p_basic(x))
                sumC += herdsman.p_basic(x)
                count += 1
        return (-self.l_fairself * sumA - self.l_fairother * sumB / sum(self.friendship_weights)) /\
               sumC * self.p_basic(x) if sumC is not 0.0 else 0

    def add_recip(self, x):
        """
        If what this herdsman does is similar to what other herdsmen did in the
        last timestep this will return a higher value. The social network 
        weights are also taken into account.
        """
        sumA, sumB, sumC = 0, 0, 0
        count = 0
        for herdsman in self.model.herdsmen:
            if herdsman is not self:
                sumA += self.friendship_weights[count] * self.s(-1, herdsman, -1)
                sumB += self.friendship_weights[count] * self.s(0, herdsman, -1)
                sumC += self.friendship_weights[count] * self.s(1, herdsman, -1)
                count += 1
        if x[self.index] < 0:
            return self.p_basic(x) * self.l_negrecip * (sumA + .5 * sumB) / sum(self.friendship_weights)
        elif x[self.index] > 0:
            return self.p_basic(x) * self.l_posrecip * (sumC + .5 * sumB) / sum(self.friendship_weights)
        else:
            return self.p_basic(x) * .5 * (self.l_negrecip * (sumA + .5 * sumB) + self.l_posrecip * (sumC + .5 * sumB)) /\
                   sum(self.friendship_weights)

    def add_conf(self, x):
        """
        If what this herdsman does is similar to what other herdsmen did on 
        average during the simulation, this will return a higher value. The 
        social network weights are also taken into account.
        """
        if len(self.a) == 0:
            return 0
        sumA = 0
        count = 0
        for herdsman in self.model.herdsmen:

            if herdsman is not self:
                for t in range(len(self.a)):
                    sumA = sumA + self.friendship_weights[count] * self.s(x[self.index], herdsman, t)
                count += 1

        return self.p_basic(x) * self.l_conf * sumA / (len(self.a) * sum(self.friendship_weights))

    def decide(self):
        """
        Making a decision happens by trying out all three decisions. The
        decision with the highest payoff gets made.
        """
        x = Herdsman.x
        x[self.index] = -1
        y = np.zeros(x.shape)
        y[self.index] = -1
        a = self.p_coop(y) + self.add_fair(y) + self.add_recip(y) + self.add_conf(y)
        # If a herdsman does not have any sheep it should be impossible to
        # remove them.
        if len(self.stock) == 0:
            a = -float('inf')
        x[self.index] = 0
        y[self.index] = 0

        b = self.p_coop(y) + self.add_fair(y) + self.add_recip(y) + self.add_conf(y)
        x[self.index] = 1
        y[self.index] = 1

        c = self.p_coop(y) + self.add_fair(y) + self.add_recip(y) + self.add_conf(y)
        x[self.index] = -1 if a > b and a > c else 0 if b > c else 1

        return x[self.index]


class TotC(Model):
    """
    Main model for the tragedy of the commons
    """


    def __init__(self, initial_herdsmen=5, initial_sheep_per_herdsmen=0, initial_edges=5, l_coop=0, l_fairself=0,
                 l_fairother=0, l_negrecip=0, l_posrecip=0, l_conf=0):
        super().__init__()
        self.width = GRID_SIZE
        self.height = GRID_SIZE

        self.grass = []
        self.herdsmen = []
        self.initial_herdsmen = initial_herdsmen
        self.initial_sheep_per_herdsmen = initial_sheep_per_herdsmen
        self.sheep_survival_rate= []

        self.l_coop = l_coop
        self.l_fairself = l_fairself
        self.l_fairother = l_fairother
        self.l_negrecip = l_negrecip
        self.l_posrecip = l_posrecip
        self.l_conf = l_conf

        self.G = nx.gnm_random_graph(initial_herdsmen, initial_edges)

        Sheep.sheepdeaths = 0
        Herdsman.i = 0
        Herdsman.x = np.zeros(initial_herdsmen, dtype=np.int8)

        self.schedule_Grass = RandomActivation(self)
        self.schedule_Herdsman = StagedActivation(self, stage_list=["advance", "step"], shuffle=True)
        self.schedule_Sheep = RandomActivation(self)
        self.schedule = RandomActivation(self)

        self.grid = MultiGrid(self.width, self.height, torus=True)

        # "Grass" is the number of sheep that the grass can sustain
        self.datacollector = DataCollector(
            {"Grass": lambda m: self.get_expected_grass_growth() / .5,
             "Sheep": lambda m: self.get_sheep_count(),
             "Sheep deaths": lambda m: Sheep.sheepdeaths })

        self.init_population()

        # required for the datacollector to work
        self.running = True
        self.datacollector.collect(self)

    def add_herdsman(self):
        """
        At a herdsman at a random position on the grid.
        """
        x = random.randrange(self.width)
        y = random.randrange(self.height)
        herdsman = self.new_agent(Herdsman, (x, y))
        self.herdsmen.append(herdsman)

    def init_grass(self):
        """
        Initialise a patch of grass at every square on the grid.
        """
        for agent, x, y in self.grid.coord_iter():
            self.grass.append(self.new_agent(Grass, (x, y)))

    def init_herdsman(self):
        """
        Spawn `initial_herdsmen' herdsmen on the field.
        """
        for i in range(getattr(self, "initial_herdsmen")):
            self.add_herdsman()

    def init_node_attr(self):
        """
        Assign the unique herdsman ID as attribute to graph nodes for the social
        network.
        """
        N = self.get_herdsman_count()
        for i in range(getattr(self, "initial_herdsmen")):
            for j in range(getattr(self, "initial_herdsmen")):
                if i is not j:
                    if nx.has_path(self.G, source=i, target=j) == True:
                        if nx.shortest_path_length(self.G, source=i, target=j) == 1:
                            if sum(nx.common_neighbors(self.G, i, j)) > 5:
                                self.herdsmen[i].friendship_weights.append(1)
                            else:
                                self.herdsmen[i].friendship_weights.append(.75 + .05 * sum(nx.common_neighbors(self.G, i, j)))
                        elif nx.shortest_path_length(self.G, source=i, target=j) == 1:
                            if sum(nx.common_neighbors(self.G, i, j)) > 10:
                                self.herdsmen[i].friendship_weights.append(1)
                            else:
                                self.herdsmen[i].friendship_weights.append(.5 + .05 * sum(nx.common_neighbors(self.G, i, j)))
                        else:
                            self.herdsmen[i].friendship_weights.append(1 / nx.shortest_path_length(self.G, source=i, target=j))
                    else:
                        self.herdsmen[i].friendship_weights.append(1 / N)

    def init_population(self):
        """
        Initialise grass, herdsmen, sheep, and the social network
        """
        self.init_grass()
        self.init_herdsman()
        self.init_node_attr()

    def get_expected_grass_growth(self):
        """
        Get an estimate of the expected grass growth for the next timestep. 
        If grass is fully grown it will return 0.0123 (the average grass growth
        over its lifetime.
        """
        return sum([grass.next_density() if grass.density < 0.99 else 0.0123 for grass in self.grass])

    def get_grass_count(self):
        """
        Get a sum of all grass densities.
        """
        return sum([grass.density for grass in self.grass])

    def get_herdsman_count(self):
        return self.schedule_Herdsman.get_agent_count()

    def get_sheep_count(self):
        return self.schedule_Sheep.get_agent_count()

    def new_agent(self, agent_type, pos):
        """
        Create new agent, and add it to the scheduler.
        """
        agent = agent_type(self.next_id(), self, pos)
        self.grid.place_agent(agent, pos)
        getattr(self, f"schedule_{agent_type.__name__}").add(agent)

        return agent

    def remove_agent(self, agent):
        """
        Remove agent from the grid and the scheduler.
        """
        self.grid.remove_agent(agent)
        getattr(self, f"schedule_{type(agent).__name__}").remove(agent)

    def run_model(self, step_count=200):
        """
        Runs the model for a specific amount of steps.
        """
        for i in range(step_count):
            self.step()

    def step(self):
        """
        Calls the step method for grass and sheep.
        """
        self.schedule_Grass.step()
        a = self.get_sheep_count()
        self.schedule_Sheep.step()
        b = self.get_sheep_count()
        c = b / a if a > 0 else 0
        self.sheep_survival_rate.append(c)
        self.schedule_Herdsman.step()
        self.schedule.step()

        # save statistics
        self.datacollector.collect(self)

