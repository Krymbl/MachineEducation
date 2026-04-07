import random
from os import name

import numpy as np
import pygame
import sklearn
from matplotlib import pyplot as plt


#import numpy
# matplotlib.pyplot as plt

def main():
    data = np.array(generate())

    plt.scatter(data[:,0], data[:,1])
    plt.show()

    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    pygame.draw.circle(screen,"white", event.pos, 10)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    dbscan()
        pygame.display.flip()


def dbscan():
    pass

def generate(numder_of_group = 3, points_count = 10):
    data = []
    for _ in range(numder_of_group):
        cX, cY = random.random() * 5.0, random.random() * 5.0

        for _ in range(points_count):
            data.append(np.array(
                [random.gauss(cX, 0.5), random.gauss(cY, 0.5)]))
    return data


if __name__ == "__main__":
    main()
