# * A decorator is a method that takes another method as an argument and extends the way the original method works.
"""
We’ll use two built-in decorators and apply them to two methods in a class to implement a property.

@<decorator>
def someMethod(self, <parameters>) 

Property: 
An attribute of a class that appears to client code to be an instance variable,
but instead causes a method to be called when it's accessed. 
"""


class Student:
    def __init__(self, name, startingGrade=0):
        """
        The __init__() method first stores the student’s name in an instance variable. The next line 1 seems straightforward but is a little unusual.
        As we have seen, we typically store the values of parameters into instance variables.
        Therefore, we might be tempted to write this line as:
            self.__grade = startingGrade
        """
        self.__name = name
        self.grade = startingGrade  # ! -  Since this is a property, the self.grade is actually making a ref. to the setter.

    # ? - This one defines grade as a property of any object created from this class
    @property  # 2 - We have two methods named grade. One is the getter ->
    def grade(self):  # 3
        return self.__grade

    @grade.setter  # 4 - and this is the setter
    def grade(
        self, newGrade
    ):  # ? 5 - Accepts new value as a parameter, but does a number of checks then sets the value.
        try:
            newGrade = int(newGrade)
        except (TypeError, ValueError) as e:
            raise type(e)("New grade: " + str(newGrade) + ", is an invalid type.")
        if (newGrade < 0) or (newGrade > 100):
            raise ValueError(
                "New grade:" + str(newGrade) + ", must be between 0 and 100."
            )
        self.__grade = newGrade


oStudent1 = Student("Joe Schmoe")
oStudent2 = Student("Jane Smith")

print(
    oStudent1.grade
)  # * Here it looks like we're reaching out directly, but since grade is a property, it refers to the property
print(oStudent2.grade)  # * Returns the value of the private variable __grade
print()

oStudent1.grade = 85  # * Here we are setting. It looks like we're setting values directly but grade is a property. Python turns these into calls
oStudent2.grade = 92  # * to the setter method. Validating before each assignment.

print(oStudent1.grade)
print(oStudent2.grade)

"""
Using the @property and @<property_name>.setter decorators gives you the best of both the direct access and getter-and-setter worlds.
PIEA -> Polymorphism -> Inheritance -> Encapsulation -> Abstraction (Handling complexity by hiding unnecessary details)
"""
