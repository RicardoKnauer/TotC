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
                    agent.fade()
                    grass_eaten = True
                    break
        # valid choice???
        if not grass_eaten:
            self.random_move()


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
        self.density += 0.05 * (1 - self.density) * self.density


class Herdsman(Agent):

    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)

        self.pos = pos
        self.unique_id = unique_id


class TotC(Model):

    def __init__(self, width=33, height=33, initial_herdsmen=5, initial_sheep_per_herdsmen=5):
        super().__init__()
        self.height = width
        self.width = height
        self.initial_herdsmen = initial_herdsmen
        self.initial_sheep_per_herdsmen = initial_sheep_per_herdsmen
        self.stock = {}

        self.schedule_Grass = RandomActivation(self)
        self.schedule_Herdsman = RandomActivation(self)
        self.schedule_Sheep = RandomActivation(self)

        self.grid = MultiGrid(self.width, self.height, torus=True)
        self.datacollector = DataCollector(
            {"Grass": lambda m: self.schedule_Grass.get_agent_count(),
             "Sheep": lambda m: self.schedule_Sheep.get_agent_count()})

        self.init_population(Grass)
        self.init_population(Herdsman)

        # required for the datacollector to work
        self.running = True
        self.datacollector.collect(self)

    def init_population(self, agent_type, n=0):
        """
        Initialize population.
        """
        if agent_type is Grass:
            for agent, x, y in self.grid.coord_iter():
                self.new_agent(agent_type, (x, y))
        elif agent_type is Herdsman:
            for i in range(getattr(self, "initial_herdsmen")):
                x = random.randrange(self.width)
                y = random.randrange(self.height)
                self.new_agent(agent_type, (x, y))


    def new_agent(self, agent_type, pos):
        """
        Create new agent, and add it to the scheduler.
        """
        agent = agent_type(self.next_id(), self, pos)
        if agent_type is Herdsman:
            self.stock[getattr(agent, "unique_id")] = []
            for i in range(getattr(self, "initial_sheep_per_herdsmen")):
                x = random.randrange(self.width)
                y = random.randrange(self.height)
                sheep_id = self.new_agent(Sheep, (x, y))
                self.stock[getattr(agent, "unique_id")].append(sheep_id)
        self.grid.place_agent(agent, pos)
        getattr(self, f"schedule_{agent_type.__name__}").add(agent)

        return getattr(agent, "unique_id")

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
