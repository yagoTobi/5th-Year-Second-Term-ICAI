class TV:
    # * Okay, planning time. What do we have here?
    # * Brand
    # * Volume setting (1 to 100)
    # * Current volume setting
    # * Mute state
    # * List of available channels
    # * Current channel setting
    # * Actions:
    # * Power on/off
    # * Raise/Lower the volum
    # * Change the channel up and down
    # * Mute/Unmute the TV
    # * Get info on the current settings
    # * Go to a specified channel
    def __init__(self, brand, location):
        self.brand = brand
        self.location = location
        self.isOn = False
        self.isMuted = False
        self.channelList = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.nChannels = len(self.channelList)
        self.channelIndex = 0
        self.VOL_MIN = 0
        self.VOL_MAX = 100
        self.volume = self.VOL_MAX

    def power(self):
        self.isOn = not self.isOn  # toggle between on and off

    def volumeUp(self):
        if self.isOn == True:  # If the TV is turned off nothing
            if self.isMuted == True:
                self.isMuted = False
            if self.volume < self.VOL_MAX:
                self.volume = self.volume + 1

    def volumeDown(self):
        if self.isOn == True:
            if self.isMuted == True:
                self.isMuted = False
            if self.volume > self.VOL_MIN:
                self.volume = self.volume - 1

    def channelUp(self):
        if self.isOn == True:
            self.channelIndex = self.channelIndex + 1
            if self.channelIndex > self.nChannels:
                self.channelIndex = (
                    0  # When you exceed return to the beginning of the list
                )

    def channelDown(self):
        if self.isOn == True:
            self.channelIndex = self.channelIndex - 1
            if self.channelIndex < 0:
                self.channelIndex = self.nChannels - 1

    def mute(self):
        if self.isOn == True:
            self.isMuted = not self.isMuted

    def setChannel(self, channel_number):
        if channel_number in self.channelList:
            # ? - In the case that the channel exists, set the index to the index of the number defined on the list.
            self.channelIndex = self.channelList.index(channel_number)

    def showInfo(self):
        print()
        print("TV Status:\n")
        print(f"TV Brand: {self.brand}")
        print(f"TV Location: {self.location}")
        print(f"Power? {self.isOn}")

        if self.isOn:
            print(f"Current channel: {self.channelList[self.channelIndex]}")
            if self.isMuted:
                print(f"Volume at: {self.volume} (Sound is muted)")
            else:
                print(f"Volume at: {self.volume}")


# ? - Testing
oTV = TV("Sony")
oTV.power()
oTV.showInfo()
# * When you pass the instantiated object, that's essentially passing the self. side of it.
oTV.volumeUp()
for i in range(50):
    oTV.volumeDown()
oTV.channelUp()
oTV.showInfo()

oTV.mute()
oTV.showInfo()

oTV.setChannel(8)
oTV.mute()

oTV.showInfo()

oTV.power()
oTV.showInfo()
