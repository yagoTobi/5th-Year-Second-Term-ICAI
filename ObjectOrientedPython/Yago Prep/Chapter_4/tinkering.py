from BankAccount import *

accountsList = []
accountsDict = {}
accountId = 0
oNewAccount = BankAccount('Yago', 'magic')
oNewAccount.depositMoney(500,'magic')
oNewAccount.showBalance('magic')

accountsList.append(oNewAccount)
accountsDict[accountId] = oNewAccount
accountId += 1

# * No storage -> But we can see now that once we added a library, a folder named __pycache__ was created. 
# * This saves a compiled version of the code of that class pyc -> Python Compiled

print()
userName = input("Hi user! What's the name you want for this account?")
userBalance = int(input('What is your starting balance for this account?'))
userPassword = input('Make sure no one is looking. What password do you wish to set?')

oNewAccount = BankAccount(userName, userPassword)
oNewAccount.depositMoney(userBalance, userPassword)
oNewAccount.showBalance(userPassword)

accountsList.append(oNewAccount)
accountsDict[accountId] = oNewAccount
accountId += 1
# * We can recycle the same variable for it to occupy less storage

accountsList[0].show()
accountsList[1].show()

# If we then wish to perform a specific number of operations under a specific account 
oAccount3 = accountsList[1]
oAccount3.depositMoney(400, userPassword)


# * Interface -> The collection of methods which a class provides. It shows what an object created from the class can do 
# * Implementation -> The actual code of the class, which shows how an object does what it does. 