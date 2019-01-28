import random
from mesa import Agent
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.time import RandomActivation

import networkx as nx
from mesa.space import NetworkGrid
import matplotlib.pyplot as plt

GROWTH_RATE = 1.03 # Grass regrowth rate
REQU = 1 # Amount of grass a sheep needs to eat in one timestep
MAX_STEPS = 5 # Maximum amount of steps a sheep can do in one timestep
P = 133
GRIDSIZE = 33

def g(grass):
    return grass * (1 + (GROWTH_RATE * grass * (1 - grass / (GRIDSIZE ** 2))))

class RandomWalker(Agent):

    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)

        self.pos = pos
        self.unique_id = unique_id

    def random_move(self):
        """
        Move agent randomly to Moore's neighborhood.
        """
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        new_position = random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)


class Sheep(RandomWalker):

    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model, pos)
        self.saturation = 1
        self.owner = None

    def step(self):
        """
        Sheep either eats grass or moves randomly to Moore's neighborhood.
        """
        saturation, i = 0, 0
        while i < MAX_STEPS and saturation < REQU:
            this_cell = self.model.grid.get_cell_list_contents([self.pos])
            grass_eaten = False
            for agent in this_cell:
                if isinstance(agent, Grass):
                    saturation += agent.fade()
            self.random_move()
            i += 1
        if saturation < REQU:
            self.die()

    def die(self):
        self.model.remove_agent(self)
        self.owner.stock.remove(self)


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
        self.density = 0.1
        return tmp

    def step(self):
        """
        Grass regrows logistically.
        """
        self.density = min(1, g(self.density))


class Herdsman(Agent):

    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)

        self.pos = pos
        self.unique_id = unique_id
        self.stock = []

        self.l_coop = 1/7
        self.l_fairself = 1/7
        self.l_fairother = 1/7
        self.l_negrecip = 1/7
        self.l_posrecip = 1/7
        self.l_conf = 1/7
        self.l_risk = 1/7
        # Decision at each timestep
        self.a = []
        # Number of cattle owned at each timestep
        self.k = []
        
        # 3 is random init to see if it works
        self.degree = 3
        self.betwcent = 3
        # lists for shortest paths and common neighbours
        self.shortest_paths = []
        self.common_nbrs = []


        for i in range(model.initial_sheep_per_herdsmen):
            self.add_sheep()


    def add_sheep(self):
        x = random.randrange(self.model.width)
        y = random.randrange(self.model.height)
        sheep = self.model.new_agent(Sheep, (x, y))
        self.stock.append(sheep)
        sheep.owner = self


    def remove_sheep(self):
        sheep = self.stock.pop()
        self.model.remove_agent(sheep)


    def step(self):
        decision = self.decide()
        if decision > 0:
            self.add_sheep()
        elif decision < 0:
            self.remove_sheep()

        self.a.append(decision)


    def decide(self):
        # x is the consequence of the action
        # it can be -1 (negative), 0 (neutral), or 1 (positive)
        # (as described in Schindler)
        a = self.p_coop(-1) + self.add_fair(-1) + self.add_recip(-1) + self.add_conf(-1) + self.add_risk(-1)
        b = self.p_coop(0) + self.add_fair(0) + self.add_recip(0) + self.add_conf(0) + self.add_risk(0)
        c = self.p_coop(1) + self.add_fair(1) + self.add_recip(1) + self.add_conf(1) + self.add_risk(1)
        return -1 if a > b and a > c else 0 if b > c else 1

    def p_coop(self, x):
        basicsum = 0
        for herdsman in self.model.herdsmen:
            basicsum = basicsum + herdsman.p_basic(x) * herdsman.l_coop

        return (1 - self.l_coop) * self.p_basic(x) + basicsum

    def p_basic(self, x):
        N = self.model.get_herdsman_count()
        grass = self.model.get_grass_count()
        sheep = self.model.get_sheep_count()

        cost = g(max(0, grass - sheep * REQU)) - g(max(0, grass - (sheep + x) * REQU)) * P / REQU
        return (len(self.k) + x) * P - (cost / N)

    def add_fair(self, x):
        sumA, sumB, sumC = 0, 0, 0
        for herdsman in self.model.herdsmen:
            if herdsman is not self:
                sumA = sumA + max(herdsman.p_basic(x) - self.p_basic(x), 0)
                sumB = sumB + max(0, self.p_basic(x) - herdsman.p_basic(x))
                sumC = sumC + herdsman.p_basic(x)
        return (-self.l_fairself * sumA - self.l_fairother * sumB) / sumC * self.p_basic(x) if sumC > 0 else 0

    def add_recip(self, x):
        N = self.model.get_herdsman_count()
        sumNeg, sumNeut, sumPos = 0, 0, 0
        for herdsman in self.model.herdsmen:
            if herdsman is not self:
                sumNeg  = sumNeg  + self.s(-1, herdsman, -1)
                sumNeut = sumNeut + self.s( 0, herdsman, -1)
                sumPos  = sumPos  + self.s( 1, herdsman, -1)

        if x < 0:
            return self.p_basic(x) * self.l_negrecip * (sumNeg + 0.5 * sumNeut) / (N-1)
        elif x > 0:
            return self.p_basic(x) * self.l_posrecip * (sumPos + 0.5 * sumNeut) / (N-1)
        else:
            return self.p_basic(x) * 0.5 * (self.l_negrecip * (sumNeg + 0.5 * sumNeut) + self.l_posrecip * (sumPos + 0.5 * sumNeut)) / (N-1)

    # d : { -1, 0, 1 } (in Schindler: neg, neut, pos)
    # j: herdsman
    # t: timestep
    def s(self, d, h, t):
        return 1 if len(h.a) > 0 and h.a[t] is d else 0

    def add_conf(self, x):
        if len(self.a) == 0:
            return 0
        N = self.model.get_herdsman_count()
        sumA = 0
        for herdsman in self.model.herdsmen:
            for t in range(len(self.a)):
                sumA = sumA + self.s(x, herdsman, t)

        return self.p_basic(x) * self.l_conf * sumA / (len(self.a) * N)

    def add_risk(self, x):
        return 0


class TotC(Model):

    def __init__(self, initial_herdsmen=5, initial_sheep_per_herdsmen=5, initial_edges=5):
        super().__init__()
        self.width = GRIDSIZE
        self.height = GRIDSIZE
        self.initial_herdsmen = initial_herdsmen
        self.initial_sheep_per_herdsmen = initial_sheep_per_herdsmen
        self.herdsmen = []
        self.grass = []
        
        self.G = nx.gnm_random_graph(initial_herdsmen, initial_edges)
        # or G = nx.erdos_renyi_graph(n, p)
        self.edgelist = self.G.edges
        self.nodelist = self.G.nodes
        self.unique_id_list = []
        self.shortest_paths_list = []


        self.schedule_Grass = RandomActivation(self)
        self.schedule_Herdsman = RandomActivation(self)
        self.schedule_Sheep = RandomActivation(self)

        self.grid = MultiGrid(self.width, self.height, torus=True)
        # Grass is actually number of sheep grass can support
        self.datacollector = DataCollector(
            {"Grass": lambda m: self.get_grass_count() * (GROWTH_RATE - 1),
             "Sheep": lambda m: self.get_sheep_count()})

        self.init_population()

        # required for the datacollector to work
        self.running = True
        self.datacollector.collect(self)

    def get_grass_count(self):
        return sum([grass.density for grass in self.grass])

    def get_herdsman_count(self):
        return self.schedule_Herdsman.get_agent_count()

    def get_sheep_count(self):
        return self.schedule_Sheep.get_agent_count()


    def init_population(self):
        self.init_grass()
        self.init_herdsman()
        self.init_node_attr()
        self.init_herds_attr()


    def init_grass(self):
        for agent, x, y in self.grid.coord_iter():
            self.grass.append(self.new_agent(Grass, (x, y)))


    def init_herdsman(self):
        for i in range(getattr(self, "initial_herdsmen")):
            self.add_herdsman()


    def add_herdsman(self):
        x = random.randrange(self.width)
        y = random.randrange(self.height)
        herdsman = self.new_agent(Herdsman, (x, y))
        self.herdsmen.append(herdsman)
        # get list of herdsman IDs
        self.unique_id_list.append(herdsman.unique_id)

    def get_herdsman(self, j):
        return self.herdsmen[j]
    
    # giving the nodes of the graph the unique herdsman IDs as attribute
    def init_node_attr(self):
        for i in range(getattr(self, "initial_herdsmen")):
            self.G.nodes[i]['herds_id'] = self.unique_id_list[i]
           # self.G.nodes[i] = self.unique_id_list[i]
            for j in range(getattr(self, "initial_herdsmen")):
                if i is not j:
                    if nx.has_path(self.G, source=i, target=j) == True:
                        self.shortest_paths_list.append(1 / nx.shortest_path_length(self.G, source=i, target=j))
                        self.herdsmen[i].shortest_paths.append(1 / nx.shortest_path_length(self.G, source=i, target=j))
                    else:
                        self.shortest_paths_list.append(1/10)
                        self.herdsmen[i].shortest_paths.append(1/10)

    # giving the initialized herdsman the graph attributes
    def init_herds_attr(self):
        for herdsman in self.herdsmen:
            for i in range(getattr(self, "initial_herdsmen")):
                if self.G.nodes[i]['herds_id'] == herdsman.unique_id:
                    herdsman.degree = self.G.degree[i]




    def new_agent(self, agent_type, pos):
        """
        Create new agent, and add it to the scheduler.
        """
        agent = agent_type(self.next_id(), self, pos)
        self.grid.place_agent(agent, pos)
        getattr(self, f"schedule_{agent_type.__name__}").add(agent)

        return agent

    # not yet needed...
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
        self.schedule_Sheep.step()
        self.schedule_Herdsman.step()

        # save statistics
        self.datacollector.collect(self)


test=TotC()
test.run_model()
print(test.height)
print(test.nodelist)
print(test.edgelist)
print(test.G[0])
print(test.shortest_paths_list)
print(len(test.shortest_paths_list))
print(test.herdsmen[1].shortest_paths)
nx.draw(test.G, pos=nx.circular_layout(test.G), nodecolor='r', edgecolor= 'b')
plt.show()