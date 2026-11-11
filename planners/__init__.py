"""Public planner factory; importing it never starts model calls."""
from .base import BasePlanner
from .rules import FIFOPlanner, EDFPlanner, GreedyPlanner
from .ortools_planner import ORToolsPlanner
from .llm import LLMPlanner, LLMNotConfigured
from .learned import LearnedPlanner, LearnedNotConfigured


def make_planner(name, config=None):
    classes = {'fifo': FIFOPlanner, 'edf': EDFPlanner, 'greedy': GreedyPlanner,
               'ortools': ORToolsPlanner, 'llm': LLMPlanner, 'learned': LearnedPlanner}
    if name not in classes:
        raise ValueError(f'unknown planner: {name}')
    return classes[name](config)
