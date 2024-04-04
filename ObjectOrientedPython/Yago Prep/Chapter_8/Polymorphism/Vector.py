import math


class Vector:
    """The Vector class represents two values as a vector, allowing for many math calcs"""

    def __init__(self, x, y):
        self.__x = x
        self.__y = y

    @property  # Getter
    def x(self):
        return self.__x

    @x.setter
    def x(self, value):
        self.__x = value

    @property  # Getter
    def y(self):
        return self.__y

    @y.setter
    def y(self, value):
        self.__y = value

    def __add__(self, oOther):
        return Vector(self.x + oOther.x, self.y + oOther.y)

    def __sub__(self, oOther):
        return Vector(self.x - oOther.x, self.y - oOther.y)

    def __mul__(self, oOther):
        # * Multiplying by a vector
        if isinstance(oOther, Vector):
            return Vector((self.x * oOther.x), (self.y * oOther.y))
        # * Multiplying by a scalar
        elif isinstance(oOther, (int, float)):
            return Vector((self.x * oOther), (self.y * oOther))
        else:
            raise TypeError("Second value must be a vector or scalar")

    def __abs__(self):  # * To get the magnitude, the absolute value
        return math.sqrt((self.x**2) + (self.y**2))

    def __eq__(self, oOther):  # Called for the == operator
        return (self.x == oOther.x) and (self.y == oOther.y)

    def __gt__(self, oOther):
        if abs(self) > abs(oOther):  # * Calls the absolute method
            return True
        else:
            return False

    def __str__(
        self,
    ):  # * String representation to showcase the object and make it easily readable.
        return f"[{self.x}, {self.y}]"


oVector1 = Vector(3, 2)
oVector2 = Vector(1, 3)

oNewVector = oVector1 * oVector2

print(oNewVector)
