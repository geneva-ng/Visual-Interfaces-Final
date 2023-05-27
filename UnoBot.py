import cv2
from ultralytics import YOLO

def card_to_array(card):
    # Initialize the array with None values
    arr = [None, None, None]

    # Handle special cases first
    if card == "draw_four":
        return ['wild', None, 'draw4']
    if card == "wild_card":
        return ['wild', None, None]

    # For other cases, split the card string by "_"
    parts = card.split("_")

    # The first part is the color
    arr[0] = parts[0]

    # The second part is either a number or an action
    if parts[1].isdigit():
        arr[1] = parts[1]
    else:
        arr[1] = None
        if parts[1] == "rev":
            arr[2] = "reverse"
        elif parts[1] == "skip":
            arr[2] = "skip"
        elif parts[1] == "draw":
            arr[2] = "draw2"

    return arr

# CARD SELECTION ALGORITHM
def get_possible_hand(hand, deck):
    color = deck[0]
    number = deck[1]
    action = deck[2]
    possible_hand = []

    for card in hand:
        card_color = card[0]
        card_number = card[1]
        card_action = card[2]
        if color == 'wild':
            possible_hand.append(card)
        elif action == 'draw2':
            return 'draw2'
        elif action == 'skip':
            return 'skip'
        elif card_color == color or (card_number == number and card_number != None) or (
                card_action == action and card_action != None) or card_color == "wild":
            possible_hand.append(card)

    if not possible_hand:  # Check if the list is empty
        return None

    choice = possible_hand[0]
    return choice

key = cv2.waitKey(1)
deckcam = cv2.VideoCapture(0)
selfiecam = cv2.VideoCapture(1)

while True:
    try:
        check1, frame1 = deckcam.read()
        check2, frame2 = selfiecam.read()
       
        cv2.imshow("Capturing 1", frame1)
        cv2.imshow("Capturing 2", frame2)

        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite(filename='deck.jpg', img=frame1)
            cv2.imwrite(filename='hand.jpg', img=frame2)
            deckcam.release()
            selfiecam.release()
            cv2.waitKey(1650)
            cv2.destroyAllWindows()
            print("Image saved!")
            break

        elif key == ord('q'):
            print("Turning off camera.")
            deckcam.release()
            selfiecam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break

    except(KeyboardInterrupt):
        print("Turning off camera.")
        deckcam.release()
        selfiecam.release()       
        print("Program ended.")
        cv2.destroyAllWindows()
        break


# Load the local YOLO model
model = YOLO("best_2.pt")

# Class names
classNames = ["blue_0","blue_1","blue_2", "blue_3", "blue_4", "blue_5", "blue_6", "blue_7", "blue_8", "blue_draw", "blue_rev", "blue_skip", "draw_four", "green_0","green_1","green_2", "green_3", "green_4", "green_5", "green_6", "green_7", "green_8", "green_draw", "green_rev", "green_skip", "red_0","red_1","red_2", "red_3", "red_4", "red_5", "red_6", "red_7", "red_8", "red_draw", "red_rev", "red_skip", "wild_card", "yellow_0","yellow_1","yellow_2", "yellow_3", "yellow_4", "yellow_5", "yellow_6", "yellow_7", "yellow_8", "yellow_draw", "yellow_rev", "yellow_skip"]

deck_results = model.predict("deck.jpg")
hand_results = model.predict("hand.jpg")

deck_raw = [] 
hand_raw = []

for r in deck_results:
    boxes = r.boxes 
    for box in boxes:
        cls = int(box.cls[0])
        cardDetected = classNames[cls]
        # print(cardDetected)
        deck_raw.append(cardDetected)

for r in hand_results:
    boxes = r.boxes 
    for box in boxes:
        cls = int(box.cls[0])
        cardDetected = classNames[cls]
        # print(cardDetected)
        hand_raw.append(cardDetected)


deck = deck_raw[0]
hand = hand_raw

deck = card_to_array(deck)
hand = [card_to_array(card) for card in hand]

print("hand", hand)
print("deck", deck)

# PRINT RESULTS 
result = get_possible_hand(hand, deck)
if result == 'skip':
    print('sorry, turn skipped')
elif result == 'draw2':
    print("sorry, draw 2 cards")
elif not result:
    print("draw another card!")
else:
    print("put down this card: ", result)
