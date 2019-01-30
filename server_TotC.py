import IPython
import os
import sys
from model_TotC import *
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule

from mesa.visualization.TextVisualization import TextData
from mesa.visualization.UserParam import UserSettableParameter

v_slider = UserSettableParameter('slider', "Number of Herdsman", 5, 1, 10, 1)
e_slider = UserSettableParameter('slider', "Number of Edges [max. (V*(V-1)/2]", 5, 0, 45, 1)
l_coop_slider = UserSettableParameter('slider', "l_coop", 1, 0, 1, .1)
l_fairself_slider = UserSettableParameter('slider', "l_fairself", 1, 0, 1, .1)
l_fairother_slider = UserSettableParameter('slider', "l_fairother", 1, 0, 1, .1)
l_negrecip_slider = UserSettableParameter('slider', "l_negrecip", 1, 0, 1, .1)
l_posrecip_slider = UserSettableParameter('slider', "l_posrecip", 1, 0, 1, .1)
l_conf_slider = UserSettableParameter('slider', "l_conf", 1, 0, 1, .1)
l_risk_slider = UserSettableParameter('slider', "l_risk", 1, 0, 1, .1)

# change stdout so most prints etc. can be ignored
orig_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
IPython.get_ipython().magic("run model_TotC.py")
sys.stdout = orig_stdout


def agent_portrayal(agent):
    if type(agent) is Herdsman:
        portrayal = {"Shape": "rect",
                     "Filled": "true"}
    else:
        portrayal = {"Shape": "circle",
                 "Filled": "true"}

    if type(agent) is Grass:
        portrayal["Color"] = "green"
        portrayal["Layer"] = 0
        if agent.density < 0.1:
            portrayal["r"] = 0.1
        elif agent.density < 0.3:
            portrayal["r"] = 0.2
        elif agent.density < 0.5:
            portrayal["r"] = 0.3
        elif agent.density < 0.7:
            portrayal["r"] = 0.4
        else:
            portrayal["r"] = 0.5

    elif type(agent) is Herdsman:
        portrayal["Color"] = "black"
        portrayal["Layer"] = 1
        portrayal["w"] = .9
        portrayal["h"] = .9
        portrayal["text"] = 'A'
        portrayal["text_color"] = "white"

    elif type(agent) is Sheep:
        portrayal["Color"] = "blue"
        portrayal["Layer"] = 2
        portrayal["r"] = 0.8

    return portrayal


# create a grid of 33 by 33 cells, and display it as 500 by 500 pixels
grid = CanvasGrid(agent_portrayal, 33, 33, 500, 500)

# create dynamic linegraph
chart = ChartModule([{"Label": "Grass",
                      "Color": "green"},
                     {"Label": "Sheep",
                      "Color": "blue"}],
                    data_collector_name='datacollector')

# create  server, and pass grid and the graphs
server = ModularServer(TotC,
                       [grid, chart],
                       "Tragedy of the Commons Model",
                       {"initial_herdsmen": v_slider,
                        "initial_edges": e_slider,
                        "l_coop": l_coop_slider,
                        "l_fairself": l_fairself_slider,
                        "l_fairother": l_fairother_slider,
                        "l_negrecip": l_negrecip_slider,
                        "l_posrecip": l_posrecip_slider,
                        "l_conf": l_conf_slider,
                        "l_risk": l_risk_slider})

server.port = 8521

server.launch()


# Terminal: ipython server_TotC.py
