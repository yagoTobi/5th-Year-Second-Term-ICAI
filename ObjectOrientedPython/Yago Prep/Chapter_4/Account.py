# Account class
class AbortTransaction(Exception):
    """Raise this exception to abort a bank transaction."""

    pass


class Account:
    def __init__(self, name, balance, password):
        self.name = name
        self.balance = int(balance)
        self.password = password

    def validateAmount(self, amount):
        try:
            amount = int(amount)
        except ValueError:
            raise AbortTransaction("Amount must be an integer")

        if amount <= 0:
            raise AbortTransaction("Amount must be positive")

        return amount

    def checkPasswordMatch(self, password):
        if password != self.password:
            raise AbortTransaction("Incorrect password for this account")

    def deposit(self, amountToDeposit, password):
        if password != self.password:
            print("Sorry, incorrect password")
            return None

        if amountToDeposit < 0:
            print("You cannot deposit a negative amount")
            return None

        self.balance = self.balance + amountToDeposit
        return self.balance

    def withdraw(self, amountToWithdraw, password):
        if password != self.password:
            print("Incorrect password for this account")
            return None

        if amountToWithdraw < 0:
            print("You cannot withdraw a negative amount")
            return None

        if amountToWithdraw > self.balance:
            print("You cannot withdraw more than you have in your account")
            return None

        self.balance = self.balance - amountToWithdraw
        return self.balance

    def getBalance(self, password):
        if password != self.password:
            print("Sorry, incorrect password")
            return None
        return self.balance

    # Added for debugging
    def show(self):
        print("       Name:", self.name)
        print("       Balance:", self.balance)
        print("       Password:", self.password)
        print()
