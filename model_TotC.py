import random
from mesa import Agent
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.time import RandomActivation, BaseScheduler

import numpy as np
import networkx as nx
from mesa.space import NetworkGrid
import matplotlib.pyplot as plt

GROWTH_RATE = 0.0496 # Grass regrowth rate
REQU = 1 # Amount of grass a sheep needs to eat in one timestep
MAX_STEPS = 50 # Maximum amount of steps a sheep can do in one timestep
P = 133
GRIDSIZE = 33

def g(grass):
    #return grass * (1 + (GROWTH_RATE * grass * (1 - (grass / 1089))))
    return grass + (GROWTH_RATE * grass * (1089 - grass) / 1089)

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
                    if agent.density > 0.75:
                        agent.fade()
                        saturation += 1
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
        self.density += self.next_density()

    def next_density(self):
        return GROWTH_RATE * self.density * (1 - self.density)

class Herdsman(Agent):
    x = None
    i = 0

    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)

        self.pos = pos
        self.unique_id = unique_id
        self.stock = []

        self.l_coop = 0
        self.l_fairself = 1
        self.l_fairother = 1
        self.l_negrecip = 0
        self.l_posrecip = 0
        self.l_conf = 0
        self.l_risk = 0
        # Decision at each timestep
        self.a = []
        # Number of cattle owned at each timestep
        self.k = []
        self.index = Herdsman.i
        Herdsman.i += 1
        print('new herdsman born')
        # 3 is random init to see if it works
        self.degree = 3

        self.friendship_weights = []



        for i in range(model.initial_sheep_per_herdsmen):
            self.add_sheep()


    def add_sheep(self):
        x = random.randrange(self.model.width)
        y = random.randrange(self.model.height)
        sheep = self.model.new_agent(Sheep, (x, y))
        self.stock.append(sheep)
        sheep.owner = self


    def remove_sheep(self):
        if len(self.stock) == 0:
            return
        sheep = self.stock.pop()
        self.model.remove_agent(sheep)

    def advance(self):
        self.decision = self.decide()
        print(Herdsman.x)

    def step(self):
        if self.decision > 0:
            self.add_sheep()
        elif self.decision < 0:
            self.remove_sheep()

        self.a.append(self.decision)
        Herdsman.x[:] = 0


    def decide(self):
        # x is the consequence of the action
        # it can be -1 (negative), 0 (neutral), or 1 (positive)
        # (as described in Schindler)
        x = Herdsman.x

        x[self.index] = -1
        a = self.p_coop(x) + self.add_fair(x) + self.add_recip(x) + self.add_conf(x) + self.add_risk(x)
        print('-1 %d' % self.p_coop(x))
        x[self.index] = 0
        b = self.p_coop(x) + self.add_fair(x) + self.add_recip(x) + self.add_conf(x) + self.add_risk(x)
        print('0 %d' % self.p_coop(x))
        x[self.index] = 1
        c = self.p_coop(x) + self.add_fair(x) + self.add_recip(x) + self.add_conf(x) + self.add_risk(x)
        print('1 %d' % self.p_coop(x))
        print(a, b, c)

        x[self.index] = -1 if a > b and a > c else 0 if b > c else 1
        return x[self.index]

    def p_coop(self, x):
        basicsum = 0
        count = 0
        for herdsman in self.model.herdsmen:
            if herdsman is not self:
                basicsum = basicsum + self.friendship_weights[count] * herdsman.p_basic(x)
                count += 1
        return (1 - self.l_coop) * self.p_basic(x) + basicsum * self.l_coop / sum(self.friendship_weights)

    def p_basic(self, x):
        N = self.model.get_herdsman_count()
        grass = self.model.get_grass_count()
        sheep = self.model.get_sheep_count()

        cost = (g(max(0, grass - sheep * REQU)) - g(max(0, grass - (sheep + x.sum()) * REQU))) * P / REQU
        return (len(self.k) + x[self.index]) * P - (cost / N)

    def add_fair(self, x):
        sumA, sumB, sumC = 0, 0, 0
        count = 0
        for herdsman in self.model.herdsmen:
            if herdsman is not self:
                sumA = sumA + max(herdsman.p_basic(x) - self.p_basic(x), 0)
                sumB = sumB + self.friendship_weights[count] * max(0, self.p_basic(x) - herdsman.p_basic(x))
                sumC = sumC + herdsman.p_basic(x)
                count += 1
        return (-self.l_fairself * sumA - self.l_fairother * sumB / sum(self.friendship_weights)) / sumC * self.p_basic(x) if sumC > 0 else 0

    def add_recip(self, x):
        N = self.model.get_herdsman_count()
        sumNeg, sumNeut, sumPos = 0, 0, 0
        count = 0
        for herdsman in self.model.herdsmen:
            if herdsman is not self:
                sumNeg  = sumNeg  + self.friendship_weights[count] * self.s(-1, herdsman, -1)
                sumNeut = sumNeut + self.friendship_weights[count] * self.s( 0, herdsman, -1)
                sumPos  = sumPos  + self.friendship_weights[count] * self.s( 1, herdsman, -1)
                count += 1

        if x[self.index] < 0:
            return self.p_basic(x) * self.l_negrecip * (sumNeg + 0.5 * sumNeut) / sum(self.friendship_weights)
        elif x[self.index] > 0:
            return self.p_basic(x) * self.l_posrecip * (sumPos + 0.5 * sumNeut) / sum(self.friendship_weights)
        else:
            return self.p_basic(x) * 0.5 * (self.l_negrecip * (sumNeg + 0.5 * sumNeut) + self.l_posrecip * (sumPos + 0.5 * sumNeut)) / sum(self.friendship_weights)

    # d : { -1, 0, 1 } (in Schindler: neg, neut, pos)
    # h: herdsman
    # t: timestep
    def s(self, d, h, t):
        return 1 if len(h.a) > 0 and h.a[t] is d else 0

    def add_conf(self, x):
        if len(self.a) == 0:
            return 0
        N = self.model.get_herdsman_count()
        sumA = 0
        count = 0
        for herdsman in self.model.herdsmen:
            if herdsman is not self:
                for t in range(len(self.a)):
                    sumA = sumA + self.friendship_weights[count] * self.s(x, herdsman, t)
                count += 1
        return self.p_basic(x) * self.l_conf * sumA / (len(self.a) * sum(self.friendship_weights))

    def add_risk(self, x):
        return 0


class RandomSimultaneousActivation(BaseScheduler):
    def step(self):
        random.shuffle(self.agents)
        for agent in self.agents[:]:
            agent.advance()
        for agent in self.agents[:]:
            agent.step()
        self.steps += 1
        self.time += 1


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
        Herdsman.x = np.zeros(initial_herdsmen)
        Herdsman.i = 0
        # or G = nx.erdos_renyi_graph(n, p)
        self.edgelist = self.G.edges
        self.nodelist = self.G.nodes
        self.unique_id_list = []

        self.schedule_Grass = RandomActivation(self)
        self.schedule_Herdsman = RandomSimultaneousActivation(self)
        self.schedule_Sheep = RandomActivation(self)

        self.grid = MultiGrid(self.width, self.height, torus=True)
        # Grass is actually number of sheep grass can support
        self.datacollector = DataCollector(
            {"Grass": lambda m: self.get_expected_grass_growth() / REQU / 0.65,
             "Sheep": lambda m: self.get_sheep_count()})

        self.init_population()
        # required for the datacollector to work
        self.running = True
        self.datacollector.collect(self)

    # Expected grass growth
    def get_expected_grass_growth(self):
        return sum([grass.next_density() for grass in self.grass])

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
                        if nx.shortest_path_length(self.G, source=i, target=j) == 1:
                            if sum(nx.common_neighbors(self.G, i, j)) > 5:
                                self.herdsmen[i].friendship_weights.append(1)
                            else:
                                self.herdsmen[i].friendship_weights.append(0.75 + 0.05 * sum(nx.common_neighbors(self.G, i, j)))
                        elif nx.shortest_path_length(self.G, source=i, target=j) == 1:
                            if sum(nx.common_neighbors(self.G, i, j)) > 10:
                                self.herdsmen[i].friendship_weights.append(1)
                            else:
                                self.herdsmen[i].friendship_weights.append(0.5 + 0.05 * sum(nx.common_neighbors(self.G, i, j)))
                        else:
                            self.herdsmen[i].friendship_weights.append(1 / nx.shortest_path_length(self.G, source=i, target=j))
                    else:
                        self.herdsmen[i].friendship_weights.append(0.1)

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


#test=TotC()
#test.run_model()
#nx.draw(test.G, pos=nx.circular_layout(test.G), nodecolor='r', edgecolor= 'b')
#plt.show()