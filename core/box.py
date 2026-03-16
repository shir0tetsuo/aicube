import pygame
import sys
import math

pygame.init()

screen = pygame.display.set_mode((100,100), pygame.RESIZABLE)
font = pygame.font.Font(None, 20)
clock = pygame.time.Clock()

def handleInterrupts(keys:pygame.key.ScancodeWrapper):
    for event in pygame.event.get():
        if (event.type == pygame.QUIT) or keys[pygame.K_ESCAPE]:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            print(event)

def GameTick(t:int=20):
    clock.tick(20)
    dt = clock.get_time() / 1000.0
    keys = pygame.key.get_pressed()
    handleInterrupts(keys)
    return dt, keys

while True:
    
    ##########################
    dt, keys = GameTick(20)
    ##########################
    # # Free Camera Controller
    # freecam(camera, keys, dt)

    # # Debug Cube Controller
    # DebugCube.debug_control(speed, keys)
    # DebugCube.draw(project, camera, screen, font)

    ##########################
    pygame.display.flip()
    ##########################