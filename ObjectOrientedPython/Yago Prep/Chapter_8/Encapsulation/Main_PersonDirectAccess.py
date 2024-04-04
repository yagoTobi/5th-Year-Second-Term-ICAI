from Person import *

oPerson1 = Person("Joe Schmoe", 90000)
oPerson2 = Person("Jane Smith", 9900)

# * Getting the values directly, which is the big no no
# ! - This breaks the core idea of encapsulation => What are the problems?
"""
Problems with direct access: 

 Changng the name of the instance variable will break any client code which uses the name directly. 'If a var name was not appropriate. 
 So if we pass from self.originalName to self.newName, our entire system would break. 
 

 That's why we have the getters and setters, so the devs take care of the implementation but not the client. 
"""
print(oPerson1.salary)
print(oPerson2.salary)

oPerson1.salary = 100000
oPerson2.salary = 111111

print(oPerson1.salary)
print(oPerson2.salary)

oPerson1.getSalary()
oPerson1.setSalary(100000)
