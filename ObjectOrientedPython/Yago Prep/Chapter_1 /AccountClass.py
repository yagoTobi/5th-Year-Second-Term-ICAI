class Account:
    def __init__(self, name, balance, password):
        self.name = name
        self.balance = balance
        self.password = password

    def show(self):
        print(f"Name of the account: {self.name}")
        print(f"Account Balance: {self.balance}")
        print(f"Account Password: {self.password}")

    def getBalance(self, password):
        if password != self.password:
            print("Sorry, incorrect password")
            return None
        else:
            return self.balance

    def deposit(self, amountToDeposit, password):
        if password != self.password:
            print("Sorry! Incorrect password")
            return None
        elif amountToDeposit < 0:
            print("Sorry! You can't deposit a negative amount")
            return None
        else:
            self.balance = self.balance + amountToDeposit
            print(f"Success! New Balance: {self.balance}")
            return self.balance

    def withdraw(self, amountToWithdraw, password):
        if password != self.password:
            print("Sorry! Incorrect Password")
            return None
        elif amountToWithdraw < 0:
            print("You can't withdraw a negative amount")
            return None
        else:
            self.balance = self.balance - amountToWithdraw
            print(f"Amount withdrawn\nNew Balance: {self.balance}")
            return self.balance
