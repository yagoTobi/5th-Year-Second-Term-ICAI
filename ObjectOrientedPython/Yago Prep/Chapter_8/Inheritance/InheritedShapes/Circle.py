import pygame
from Shape import *
import math
from abc import ABC, abstractmethod


class Circle(Shape):
    def __init__(self, window, maxWidth, maxHeight):
        super().__init__(
            window, "Circle", maxWidth, maxHeight
        )  # * Get the defined attributes
        self.radius = random.randrange(10, 50)
        self.centerX = self.x + self.radius
        self.centerY = self.x + self.radius
        self.rect = pygame.Rect(self.x, self.y, self.radius * 2, self.radius * 2)
