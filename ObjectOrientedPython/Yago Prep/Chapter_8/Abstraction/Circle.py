"""# * We can force specific classess to be implemented like: 

import random 
from abc import ABC, abstractmethod

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

class Shape(ABC): 
    def __init__(self, window, shapeType, maxWidth, maxHeight): 
        self.window = window 
        self.shapeType = shapeType
        self.color = random.choice((RED, GREEN, BLUE))
        self.x = random.randrange(1, maxWidth - 100)
        self.y = random.randrange(1, maxHeight - 100)

    def getType(self): 
        return self.shapeType
    
    # * This is a way to force the shapes to get it
    @abstractmethod
    def clickedInside(self, mousePoint): 
        raise NotImplementedError
"""

import pygame
from Shape import *


class Circle(Shape, ABC):
    def __init__(self, window, maxWidth, maxHeight):
        super().__init__(window, "Rectangle", maxWidth, maxHeight)
        self.width = random.randrange(10, 100)
        self.height = random.randrange(10, 100)
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def clickedInside(self, mousePoint):
        clicked = self.rect.collidepoint(mousePoint)
        return clicked

    def getArea(self):
        theArea = self.width * self.height
        return theArea
