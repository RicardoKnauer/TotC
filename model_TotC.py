import random
from mesa import Agent
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.time import RandomActivation


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
        this_cell = self.model.grid.get_cell_list_contents([self.pos])
        grass_eaten = False
        for agent in this_cell:
            if isinstance(agent, Grass):
                # valid choice???
                if agent.density > 0.1:
                    self.saturation += agent.density
                    agent.fade()
                    grass_eaten = True

                    break

        # valid choice???
        if not grass_eaten:
            self.random_move()
            self.saturation -= 0.4

        if self.saturation < 0:
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
        # valid choices???
        if self.density > 0.5:
            self.density -= 0.5
        else:
            self.density = 0.001

    def step(self):
        """
        Grass regrows logistically.
        """
        # growth rate ok???
        self.density += 0.07 * (1 - self.density) * self.density


class Herdsman(Agent):

    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)

        self.pos = pos
        self.unique_id = unique_id
        self.stock = []

        self.l_coop = random.random()
        self.l_fairself = random.random()
        self.l_fairother = random.random()
        self.l_negrecip = random.random()
        self.l_posrecip = random.random()
        self.l_conf = random.random()
        self.l_risk = random.random()

        for i in range(model.initial_sheep_per_herdsmen):
            self.add_sheep()


    def add_sheep(self):
        x = random.randrange(self.model.width)
        y = random.randrange(self.model.height)
        sheep = self.model.new_agent(Sheep, (x, y))
        self.stock.append(sheep)
        sheep.owner = self


    def step(self):
        if self.should_buy_sheep():
            self.add_sheep()


    def should_buy_sheep(self):
        # x is the consequence of the action
        # it can be -1 (negative), 0 (neutral), or 1 (positive)
        # (as described in Schindler)
        x = 1
        return self.p_coop(x) + self.add_fair(x) + self.add_recip(x) + self.add_conf(x) + self.add_risk(x)

    def p_coop(self, x):
        basicsum = 0
        for herdsman in self.model.herdsmen:
            herdsman.p_basic(x)

        return (1 - self.l_coop) * self.p_basic(x) + self.l_coop * basicsum

    def p_basic(self, x):
        return self.model.get_sheep_count() / 100

    def add_fair(self, x):
        sumA, sumB, sumC = 0, 0, 0
        for herdsman in self.model.herdsmen:
            if herdsman is not self:
                sumA = sumA + max(herdsman.p_basic(x) - self.p_basic(x), 0)
                sumB = sumB + max(0, self.p_basic(x) - herdsman.p_basic(x))
                sumC = sumC + herdsman.p_basic(x)
        return (-self.l_fairself * sumA - self.l_fairother * sumB) / sumC * self.p_basic(x)

    def add_recip(self, x):
        return 0

    def add_conf(self, x):
        return 0

    def add_risk(self, x):
        return 0


class TotC(Model):

    def __init__(self, width=33, height=33, initial_herdsmen=5, initial_sheep_per_herdsmen=5):
        super().__init__()
        self.height = width
        self.width = height
        self.initial_herdsmen = initial_herdsmen
        self.initial_sheep_per_herdsmen = initial_sheep_per_herdsmen
        self.herdsmen = []
        self.grass = []

        self.schedule_Grass = RandomActivation(self)
        self.schedule_Herdsman = RandomActivation(self)
        self.schedule_Sheep = RandomActivation(self)

        self.grid = MultiGrid(self.width, self.height, torus=True)
        self.datacollector = DataCollector(
            {"Grass": lambda m: sum([grass.density for grass in self.grass]),
             "Sheep": lambda m: self.schedule_Sheep.get_agent_count()})

        self.init_population()

        # required for the datacollector to work
        self.running = True
        self.datacollector.collect(self)


    def get_herdsman_count(self):
        return self.schedule_Herdsman.get_agent_count()


    def get_sheep_count(self):
        return self.schedule_Sheep.get_agent_count()


    def init_population(self):
        self.init_grass()
        self.init_herdsman()


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
