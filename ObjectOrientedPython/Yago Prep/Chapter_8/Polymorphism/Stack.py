class Stack:
    """Stack class implements a LIFO algorithm"""

    def __init__(self, startingStackAsList=None):
        if startingStackAsList is None:
            self.dataList = []
        else:
            self.dataList = startingStackAsList[:]  # * Makes a copy

    def push(self, item):
        self.dataList.append(item)

    def pop(self):
        if len(self.dataList) == 0:
            raise IndexError
        element = self.dataList.pop()
        return element

    def peek(self):
        # Retrieve the top itm, without removing it
        item = self.dataList[-1]
        return item

    def getSize(self):
        nElements = len(self.dataList)
        return nElements

    def show(self):
        print("Stack is: ")
        for value in reversed(self.dataList):
            print(" ", value)


# ! - When it comes to polymorphism, it just refers to the fact that different objects can have the same method name.
# ! - So it's the same method, but it's implementation is different. The implementation varies with each object.

# A clear example of the polymorphism concept is the + operator. When dealing with numbers it adds, when dealing with strings it concatenates.

print(isinstance(123, int))
print(isinstance("some string", str))

value1 = 4
value2 = 5
result = value1 + value2
print(result)

value1 = "Joe"
value2 = "Schmoe"
result = value1 + " " + value2
print(result)

# * Then we also have the special methods: __<method name>__ where a lot of them are defined -> such as __init__(self, ...) all available names are available
# ! - Python calls them behind the scenes whenever it detects an operator, special function call or circumstance. They are automatic and not intended to be called by client code directly.
# ! - They call them dunder methods. Dunder => Double underscore
# ! ^^So in this case above, when we write +, the __add__() method is being called for the respective int and str class. This is helpful for when we have to determine it.
# So == is __eq__()
# != is __neq__()
# < is __lt__()
# > is __gt__() etc...


#  * Awesome, so how do we envoke this? When Python detects an == comparison where the first object is of Square class, it calls this method.
def __eq__(self, oOtherSquare):
    pass
    # * Comparison of Square objects
    # if not isinstance(oOtherSquare, Square):
    #    raise TypeError('Second object was not a Square')
    # if self.heightAndWidth == oOtherSquare.heightAndWidth: # * To allow comparison of equal method implementation
    #    return True
    # else:
    #    return False


# Python is loosely typed which doesn't require type checking the second parameter could be of any type, and that's we need to call the isInstance.
