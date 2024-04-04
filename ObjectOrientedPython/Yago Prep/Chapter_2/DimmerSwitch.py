class DimmerSwitch:
    """
    DimmerSwitch class

    Data:
        label: DimmerSwitch label to identify
        isOn: Is the Switch On or Off?
        brightness: Level of Brightness

    Methods:
        __init__
        turnOn()
        turnOff()
        raiseLevel()
        lowerLevel()
        showInfo()
    """

    def __init__(self, label):
        self.state = False
        self.bl = 0
        self.label = label

    def turnOn(self):
        self.state = True

    def turnOff(self):
        self.state = False

    def raiseLevel(self):
        # ? - No mod when off
        if self.state == True:
            if self.bl < 10:
                self.bl = self.bl + 1

    def lowerLevel(self):
        # ? - So that we don't modify it when it's off
        if self.state == True:
            if self.bl > 0:
                self.bl = self.bl - 1

    def show(self):
        print(f"Switch label: {self.label}")
        print(f"Switch is on? {self.state}")
        print(f"Brightness Level: {self.bl}")


oDimmer = DimmerSwitch("Dimmer 1")

oDimmer.turnOn()
for i in range(4):
    oDimmer.raiseLevel()
oDimmer.show()

oDimmer.turnOff()
for i in range(4):
    oDimmer.lowerLevel()
oDimmer.show()

# ? - Quick way to get through it
print("oDimmer variables:", vars(oDimmer))
