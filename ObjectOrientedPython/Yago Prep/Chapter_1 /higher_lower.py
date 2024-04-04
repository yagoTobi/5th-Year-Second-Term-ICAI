import random

SUIT_TUPLE = ("Spades", "Hearts", "Clubs", "Diamonds")
RANK_TUPLE = (
    "Ace",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "Jack",
    "Queen",
    "King",
)

NCARDS = 8


# * Pass in a deck and this function returns a random card from the deck
def getCard(deckListIn):
    thisCard = deckListIn.pop()  # * From the cards, return the top one.
    return thisCard


# * Pass in a deck and this function returns a shuffled deck
def shuffle(deckListIn):
    # ? - Make a duplicate of the initial deck
    deckListOut = deckListIn.copy()
    random.shuffle(deckListOut)
    return deckListOut


print(
    "Welcome to higher or lower.\nYou must choose whether the next card to be shown will be higher or lower than the current card."
)
print(
    "Getting it right adds 20 points, get it wrong and you lose 15.\nYou start with 50 points.\n"
)

startingDeckList = []

# ? - 1
# ? - For each type of card
for suit in SUIT_TUPLE:
    # ? - Take each possible variation of it, and add it to the array, thus giving you a complete stack of cards.
    for thisValue, rank in enumerate(RANK_TUPLE):
        cardDict = {
            "rank": rank,
            "suit": suit,
            "value": thisValue + 1,
        }  # ! - The value here is just to get the numbers from 1 - 12, without having to compare Ace, Jack, King or Queen.
        startingDeckList.append(cardDict)

    score = 50

    while True:
        print()
        # ? - 2
        gameDeckList = shuffle(
            startingDeckList
        )  # ? - Shuffle the original deck after declaring it
        currentCardDict = getCard(gameDeckList)
        currentCardRank = currentCardDict["rank"]
        currentCardValue = currentCardDict["value"]
        currentCardSuit = currentCardDict["suit"]
        print(f"Starting card is: {currentCardRank} of {currentCardSuit}")

        for cardNumber in range(0, NCARDS):
            answer = input(
                f"Will the next card be higher or lower than the {currentCardRank} of {currentCardSuit}?\n(Enter h or l):"
            )
            answer = answer.casefold()  # * In order to force the lowercase
            answer = answer[0]  # * Just to get the first character

            nextCardDict = getCard(gameDeckList)
            nextCardRank = nextCardDict["rank"]
            nextCardSuit = nextCardDict["suit"]
            nextCardValue = nextCardDict["value"]

            print(f"The next card is: {nextCardRank} of {nextCardSuit}")

            if answer == "h":
                if nextCardValue > currentCardValue:
                    score += 20
                    print(f"Correct! It was higher!")
                else:
                    score -= 15
                    print(f"Incorrect, the value was lower!")
            elif answer == "l":
                if nextCardValue < currentCardValue:
                    score += 20
                    print(f"Correct! It was lower!")
                else:
                    score -= 15
                    print(f"Incorrect, the value was higher!")

            print(f"Current Score: {score}")
            print()
            currentCardRank = nextCardRank
            currentCardValue = nextCardValue

        goAgain = input('To play again, press ENTER, or "q" to quit:')
        if goAgain == "q":
            break

        print("OK Bye! :)")
