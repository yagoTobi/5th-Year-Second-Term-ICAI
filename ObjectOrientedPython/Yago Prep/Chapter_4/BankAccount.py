global accountnum

accountnum = 0


class BankAccount:
    """
    Data:
        Client account number (A sequence)
        Client Name
        Client Password -> Don't forget
        Client Balance

    Methods:
        Check our balance
        Make a transfer
        Withdraw money
        Deposit money
    """

    def __init__(self, name, password):
        self.name = name
        self.password = password
        self.balance = 0

    def show(self):
        print(f"Account Name: {self.name}")
        print(f"Account Password: {self.password}")
        print(f"Account Balance: {self.balance}")
        print("\n\n")

    def showBalance(self, password):
        if password != self.password:
            print("Error. Incorrect password")
        else:
            print(f"Account Balance for {self.name}:\n${self.balance}")

    def makeTransfer(self, amountToTransfer, externalAccNumber, password):
        # ? - Simple verification for acc number
        if password != self.password:
            print("Error. Incorrect password")
        else:
            if externalAccNumber > 0 & externalAccNumber < 1000000:
                if amountToTransfer < self.balance:
                    self.balance = self.balance - amountToTransfer
                    print(
                        f"Transaction to Acc. Number {externalAccNumber} of ${amountToTransfer} has been successfully done."
                    )
                    print(f"Your current balance: {self.balance}")

    def withdrawMoney(self, amountToWithdraw, password):
        if amountToWithdraw > 0:
            if password != self.password:
                print("Error. Incorrect password.")
            else:
                if amountToWithdraw < self.balance:
                    self.balance = self.balance - amountToWithdraw
                    # Define method to give the user the money
                    print(
                        f"${amountToWithdraw} successfully withdrawn. Please take your money\nRemaining balance: {self.balance}"
                    )
                else:
                    print(
                        f"ERROR: Unfortunately you are exceeding your withdrawal capabilities\nAvailable balance: {self.balance}\nFeel fry to try again"
                    )
        else:
            print("Sorry, you can't withdraw a negative amount")

    def depositMoney(self, amountToDeposit, password):
        if amountToDeposit > 0:
            if password != self.password:
                print("Error. Incorrect password.")
            else:
                # * If we were to set a return variable, that's what the variable would store, but in this case it doesn't do much
                self.balance = self.balance + amountToDeposit
                print(
                    f"${amountToDeposit} correctly deposited\nNew Balance: ${self.balance}"
                )
        else:
            print("Sorry, you can't deposit a negative amount")
