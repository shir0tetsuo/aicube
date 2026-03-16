import pygame
import sys
import math
from .grid import Grid

class Matrix:
    def __init__(self):
        pygame.init()
        
        self.screen = pygame.display.set_mode((100,100), pygame.RESIZABLE)
        self.font = pygame.font.Font(None, 20)
        self.clock = pygame.time.Clock()

        self.grid = Grid(agents=[])

        self.running = True

        pass

    def handleInterrupts(self, keys:pygame.key.ScancodeWrapper):
        for event in pygame.event.get():
            if (event.type == pygame.QUIT) or keys[pygame.K_ESCAPE]:
                self.running = False
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                print(event)

    def Tick(self, t:int=20):
        self.clock.tick(20)
        dt = self.clock.get_time() / 1000.0
        keys = pygame.key.get_pressed()
        self.handleInterrupts(keys)
        return dt, keys
    
    def LOOP(self):
        while self.running:
            dt, keys = self.Tick(20)
            self.grid.render()
            pygame.display.flip()