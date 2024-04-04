# * Now they're introducing the concept of Object Managers
# * So what they are suggesting we do is to create a Bank object which will manage the different Account Objects, which makes all of the sense in the world
# * It's a list or dictionary of single class objects.
from Account import *


class Bank:
    def __init__(self):
        self.accountsDict = {}
        self.nextAccountNumber = 0

    def createAccount(self, name, startingAmount, password):
        newAccountNumber = self.nextAccountNumber
        oAccount = Account(name, startingAmount, password)
        self.accountsDict[self.nextAccountNumber] = oAccount
        self.nextAccountNumber = self.nextAccountNumber + 1
        return newAccountNumber

    def openAccount(self):
        print("*** Open Account ***")
        userName = input("What is the name for the new user account?")
        userStartingAmount = int(
            input("What is the starting balance for this account?")
        )
        userPassword = input("What is the password that you wish to set?")
        userAccountNumber = self.createAccount(
            userName, userStartingAmount, userPassword
        )  # ? - Automatically added to the dict.
        print("Your account number is:", userAccountNumber)
        print()

    def closeAccount(self):
        print("*** Close Account ***")
        userAccountNumber = int(input("What is the user account number?"))
        # ? - Verify account number in the dictionary
        userPassword = input("What is your accounts password?")
        # try except
        try:
            oAccount = self.accountsDict[userAccountNumber]
        except ValueError:
            # * ValueError, TypeError, NameError, ZeroDivision, etc...
            # * You can also raise a ValueError -> or others. This is how it works in libraries.
            # * The easiest way to do so is using inheritance, where you can define your own custom exception:
            """
            class NonFloatingPoint(Exception):
                pass # * Which contains the methods of the Exception class, but now we'll modify them
            """
            print("Please try again, that account wasn't found")

        accountBalance = oAccount.getBalance(userPassword)

        if accountBalance is not None:
            print(
                f"You have ${accountBalance} in the bank. These have been withdrawn from your account."
            )
            oAccount.withdraw(accountBalance, userPassword)
            oAccount.show()
        else:
            del self.accountsDict[userAccountNumber]
            print("Account successfully deleted")

    def balance(self):
        print("*** Get Balance***")
