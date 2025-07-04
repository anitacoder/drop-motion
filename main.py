import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8, maxHands=1)
colorR = (255, 0, 255)


class Dragrect():
    def __init__(self, posCenter, size=[200, 200]):
        self.posCenter = posCenter
        self.size = size
        self.dragging = False

    def check_collision(self, new_pos, others):
        new_cx, new_cy = new_pos
        new_w, new_h = self.size
        new_half_w = new_w // 2
        new_half_h = new_h // 2

        for other in others:
            if other is self:
                continue

            cx_other, cy_other = other.posCenter
            w_other, h_other = other.size
            half_w_other = w_other // 2
            half_h_other = h_other // 2

            x_overlap = abs(new_cx - cx_other) < (new_half_w + half_w_other)
            y_overlap = abs(new_cy - cy_other) < (new_half_h + half_h_other)

            if x_overlap and y_overlap:
                return True
        return False

    def update(self, cursor, others):
        new_pos = cursor[0], cursor[1]
        if self.dragging:
            if not self.check_collision(new_pos, others):
                self.posCenter = new_pos
        else:
            cx, cy = self.posCenter
            w, h = self.size
            if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
                if not self.check_collision(new_pos, others):
                    self.posCenter = new_pos
                    self.dragging = True


rectList = [Dragrect([x * 250 + 150, 150]) for x in range(5)]

drawing_box = False
start_point = None
end_point = None

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)

    is_any_box_dragged = False

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]

        dist_drag, _, _ = detector.findDistance(lmList[8][:2], lmList[12][:2], img)

        dist_draw, _, _ = detector.findDistance(lmList[8][:2], lmList[4][:2], img)

        cursor = lmList[8][:2]

        if dist_draw < 40:
            if not drawing_box:
                start_point = cursor
                drawing_box = True
            else:
                end_point = cursor
        else:
            if drawing_box and start_point and end_point:
                x1, y1 = start_point
                x2, y2 = end_point
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                if w > 30 and h > 30:
                    rectList.append(Dragrect([cx, cy], size=[w, h]))
            drawing_box = False
            start_point = None
            end_point = None

        if dist_drag < 60:
            for rect in rectList:
                if rect.dragging:
                    rect.update(cursor, rectList)
                    is_any_box_dragged = True
                    break

            if not is_any_box_dragged:
                for rect in rectList:
                    cx, cy = rect.posCenter
                    w, h = rect.size
                    if cx - w // 2 < cursor[0] < cx + w // 2 and \
                            cy - h // 2 < cursor[1] < cy + h // 2:
                        rect.update(cursor, rectList)
                        is_any_box_dragged = True
                        break
        else:
            for rect in rectList:
                rect.dragging = False

    else:
        for rect in rectList:
            rect.dragging = False

    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        draw_color = colorR if not rect.dragging else (0, 255, 0)
        cv2.rectangle(img, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), draw_color, cv2.FILLED)
        cvzone.cornerRect(img, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

    if drawing_box and start_point and end_point:
        cv2.rectangle(img, start_point, end_point, (0, 255, 255), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
