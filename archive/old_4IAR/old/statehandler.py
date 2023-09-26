class StateHandler:
    def verify_players(self, players):
        return True

    def get_initial_state(self, state_seed):
        return 0
    
    def get_player_turn(self, state):
        return 0

    def check_terminal_state(self, state):
        return True

    def update_state(self, state, action):
        return 0

    def get_result(self, state):
        return 0