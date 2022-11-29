import itertools
import numpy as np

from minimuse.agents.oracle import OracleAgent
from minimuse.core import constants


class ScriptAgent(OracleAgent):
    def __init__(self, env):
        super().__init__(env)
        self.scripts = iter(env.script())
        self.steps = itertools.chain(*self.scripts)

    def _compute_steps(self, obs=None):
        script = next(self.scripts, None)
        if script is None:
            self.steps = itertools.chain([])
        else:
            self.steps = itertools.chain(*script)
