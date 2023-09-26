from numbers import Number
import numpy as np

class AIPlayer():
    def __init__(self, search, constant_temperature=False):
        if constant_temperature == False:
            assert isinstance(constant_temperature, Number)
            self.get_action = lambda canonical_state, temperature: np.argmax(search.get_probs(canonical_state, temperature))
        else:
            self.get_action = lambda canonical_state: np.argmax(search.get_probs(canonical_state, temperature=0))