import sys
sys.path.append('..')

import pygame

from display import Display

class BeckDisplay(Display): 
    BLACK = (0,0,0)
    WHITE = (255,255,255)
    DARK_BROWN = (222, 184, 135)
    LIGHT_BROWN = (255, 248, 220)
    SQUARESIZE = 100
    RADIUS = int(SQUARESIZE/2 - 5)
    
    def __init__(self, game, player_names):
        self.game = game
        self.height = self.SQUARESIZE * self.game.m
        self.width = self.SQUARESIZE * self.game.n
        self.size = (self.width, self.height)
        pygame.init()
        self.screen = pygame.display.set_mode(self.size)
    
    def display_state(self, state):
        pygame.draw.rect(self.screen, self.DARK_BROWN, (0,0,self.width,self.height))
        for row in range(self.game.m):
            for col in range(self.game.n):
                colour = self.BLACK if state[row,col] == 1 else self.WHITE if state[row,col] == 2 else self.LIGHT_BROWN
                pygame.draw.circle(self.screen, colour, (int(col*self.SQUARESIZE+self.SQUARESIZE/2), int(row*self.SQUARESIZE+self.SQUARESIZE/2)), self.RADIUS)
        pygame.display.update()