# * To describe a physical object in our everyday world, we often reference attributes.
# * Desk -> color, dimensions, weight, material, etc... -> This is called state.
# * Imagine if it was like an IKEA Description of a product.  Some of these attributes are object specific, like the % of cotton on a t-shirt
# is not applicable in a car. Additionally, some objects can perform actions. -> This is called behaviour.


class LightSwitch:
    def __init__(self, switch):
        # * We could have also defined the state of the switch without passing it through the initialiser
        self.switch = switch

    # ? - These are called the methods. With always at least one parameter, self.
    # ? - __init__ is typically the first method called the constructor method.
    def turnOn(self):
        self.switch = True

    def turnOff(self):
        self.switch = False

    def show(self):
        print(self.switch)


# ? - Instantiation -> The process of creating a class.
olightSwitch1 = LightSwitch(switch=True)
olightSwitch1.turnOff()
olightSwitch1.show()

olightSwitch2 = LightSwitch(switch=False)
olightSwitch2.turnOff()
olightSwitch2.turnOn()
olightSwitch2.show()

# * Understand the relationship between a class and an object

# ? - 3 levels of scope: Local variables, global variables and "object scope".
# ? - This last one refers to all the code inside the class definition

# ? - Any variable whose name does not start with self is a local variable, and will go away when that method exists.


class Counter:
    def __init__(self) -> None:
        self.counter = 0

    def increment(self):
        self.counter = self.counter + 1

    def showValue(self):
        print(self.counter)


oCounter = Counter()
oCounter.increment()
oCounter.showValue()
# ! - 3 key differences between a function and a method:
# * All methods of a class must be indented under the class statement
# All methods have a special first parameter which is the self
# Methods in a class can use instance variables: self.var_name

# ! So no surprise here, but actually all Python data types are defined classes
