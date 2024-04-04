# * Inheritance -> Retain information from already existing classes creating a class that extends a previous one.
# ? - Base class: The class that is inherited from; which serves as a starting point for the subclass
# ? - Subclass: The class that is doing the inheriting; It enhances the base class. (Subclass inherits from a base class. )
# * Example: Student & person, car & vehicle, etc... We can redefine a method that's defined in the base class.
# * Apply the coding by difference principle
class Employee:
    def __init__(self, name, title, ratePerHour=None):
        self.__name = name
        self.__title = title
        if ratePerHour is not None:
            ratePerHour = float(ratePerHour)
        self.ratePerHour = ratePerHour

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, newName):
        self.__name = newName

    @property
    def title(self):
        return self.__title

    @title.setter
    def title(self, newTitle):
        self.__title = newTitle

    def payPerYear(self, giveBonus=False):
        pay = 52 * 5 * 8 * self.ratePerHour  # type: ignore
        return pay


class Manager(Employee):  # * Here we inherit from the Employee class
    def __init__(self, name, title, salary, reportsList=None):
        self.salary = float(salary)
        if reportsList is None:
            reportsList = []
        self.reportsList = reportsList
        super().__init__(
            name, title
        )  # * This is the equivalent of Employee.__init__(self, name, title)

    def getReports(self):
        return self.reportsList

    def payPerYear(self, giveBonus=False):  # * This overrules the previous one
        pay = self.salary
        if giveBonus:
            pay = pay + (0.10 * self.salary)  # * Bonus of 10%
            print(self.name, "gets a bonus for good work")
        return pay
