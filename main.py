import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8)
colorR = (255, 0, 255)

cx,cy, w, h = 100,100,200,200

class Dragrect():
    def __init__(self,posCenter, size=[200,200]):
        self.posCenter = posCenter
        self.size = size
    def update(self, cursor):
        cx,cy = self.posCenter
        w,h = self.size
        #if the index finger is in the rectangle region
        if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
           self.posCenter = cursor[0], cursor[1]
rectList = []
for x in range(5):
    rectList.append(Dragrect([x*250+150,150]))

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]

        l, _, _ = detector.findDistance(lmList[8][:2], lmList[12][:2], img)
        print(l)

        if l < 60:
            cursor = lmList[8] # index finger landmark
            #call the update here
            for rect in rectList:
                rect.update(cursor)

    #for the draw
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(img, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), colorR, cv2.FILLED)
        cvzone.cornerRect(img, (cx - w // 2, cy - h // 2, w, h),20,rt=0)
        cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()