# Person class
class Person:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

    # * Getters: Allow the caller to retrieve the salary
    def getSalary(self):
        return self.salary

    # * Setters: Allow the caller to set a new salary
    def setSalary(self, salary):
        self.salary = salary


class BankAccount:
    def __init__(self, balance, interestRate):
        self.balance = balance
        self.interestRate = interestRate

    def calculateInterestRate(self):
        # Assuming self.balance has been set in another method
        if self.balance < 1000:
            self.interestRate = 1.0
        elif self.balance < 5000:
            self.interestRate = 1.5
        else:
            self.interestRate = 2.0


class PrivatePerson:
    def __init__(self, name, privateData):
        self.name = name
        self.__privateData = privateData  # * Way to indicate that this data should not be accessed from outside (Oh, so this is real)

    def getName(self):
        return self.name

    def setName(self, name):
        self.name = name


oPrivatePerson = PrivatePerson("Jaquan", "this is a secrety secret")
usersPrivateData = oPrivatePerson.__privateData
# ! - This raises an error as there is no method
# ! - We can also do this in terms of methods, any attempt by client software to call the method will generate an error.
# ! - Meaning that a __method will only be used internally. => Name mangling
