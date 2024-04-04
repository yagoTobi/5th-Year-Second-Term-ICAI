# * Encapsulation -> Hiding the details but keeping them in one place.
# ? - Hides the internal details of state and behavior from any external code, and having all code in one place.
# ? - You can have direct access, getters, or setters.
# * Polymorphism -> How multiple classes can have the same methods with the same names.
# * Inheritance -> Building on top of existing code.

# ! - Encapsulation hides all the details of implementation in its methods and instance variables
# * Client -> Any software that creates an object from a class and makes calls to the method of that object.
# "Objects own their data" -> In object oriented programming, data inside an object is owned by the object.
# * Client code should only be concerned with the interface of a class and not the implementation of methods
"""A strict encapsulation says that client software should never be able to retrieve the value of an instance variable directly, only through methods."""


def calculateAverage(numbersList):
    total = 0.0
    for number in numbersList:
        total = total + number
    nElements = len(numbersList)
    average = total / nElements
    return average
