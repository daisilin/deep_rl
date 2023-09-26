class Gameplay:
    def __init__(self, players, state_handler, config, state_seed=0):
        self.players = players
        self.num_of_players = len(self.players)
        self.state_handler = state_handler
        assert self.state_handler.verify_players(self.players)
        self.state = self.state_handler.get_initial_state(state_seed)
        self.player_turn = self.state_handler.get_player_turn(self.state)

    def play(self):
        while not self.state_handler.check_terminal_state(self.state):
            action = self.players[self.player_turn].get_action(self.state)
            self.state = self.state_handler.update_state(self.state, action)
            self.player_turn = (self.player_turn + 1) % self.num_of_players
        
        return self.state_handler.get_result(self.state)
