import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8, maxHands=1)
colorR = (255, 0, 255)  # Default rectangle color


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
        else:  # Only try to pick up if not already dragging
            cx, cy = self.posCenter
            w, h = self.size
            if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
                # Only pick up if the initial move doesn't cause a collision
                if not self.check_collision(new_pos, others):
                    self.posCenter = new_pos  # Move it to current cursor position
                    self.dragging = True


rectList = [Dragrect([x * 250 + 150, 150]) for x in range(5)]

drawing_box = False
start_point = None
end_point = None  # This will be the current cursor position during drawing

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, draw=True)  # draw=True to visualize landmarks

    is_any_box_dragged = False
    cursor = None  # Initialize cursor to None

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        cursor = lmList[8][:2]  # Get index finger tip (landmark 8)

        dist_drag, _, _ = detector.findDistance(lmList[8][:2], lmList[12][:2], img)  # Index to Middle finger
        dist_draw, _, _ = detector.findDistance(lmList[8][:2], lmList[4][:2], img)  # Index to Thumb

        # --- Drawing Logic ---
        # A small distance between Index and Thumb indicates drawing intention
        if dist_draw < 40:  # If index and thumb are close (like pinching to draw)
            if not drawing_box:  # If not already drawing, start
                start_point = cursor
                drawing_box = True
            else:  # If already drawing, update the current end point
                end_point = cursor

            # Reset dragging if drawing gesture is active to prevent conflicts
            for rect in rectList:
                rect.dragging = False
            is_any_box_dragged = False  # Ensure no dragging is active while drawing

        else:  # Index and thumb are apart (drawing gesture released)
            if drawing_box and start_point and end_point:
                # Calculate the rectangle properties from start and end points
                x1, y1 = start_point
                x2, y2 = end_point

                # Ensure positive width/height by sorting points
                x_min = min(x1, x2)
                y_min = min(y1, y2)
                x_max = max(x1, x2)
                y_max = max(y1, y2)

                cx = (x_min + x_max) // 2
                cy = (y_min + y_max) // 2
                w = x_max - x_min
                h = y_max - y_min

                # Add new rectangle only if it has a reasonable size
                if w > 30 and h > 30:  # Minimum size to avoid tiny boxes
                    new_rect = Dragrect([cx, cy], size=[w, h])
                    # Check if the new box would collide with existing ones at its creation point
                    if not new_rect.check_collision(new_rect.posCenter, rectList):
                        rectList.append(new_rect)
            drawing_box = False
            start_point = None
            end_point = None  # Reset end_point as well

        # --- Dragging Logic ---
        # Only process dragging if we are NOT currently drawing a box
        if not drawing_box:
            if dist_drag < 60:  # Pinch detected (for dragging)
                # 1. Check if a box is ALREADY being dragged (priority)
                for rect in rectList:
                    if rect.dragging:
                        rect.update(cursor, rectList)
                        is_any_box_dragged = True
                        break

                # 2. If NO box is being dragged, try to start a drag
                if not is_any_box_dragged:
                    for rect in rectList:
                        # Check if cursor is inside a box to potentially pick it up
                        cx, cy = rect.posCenter
                        w, h = rect.size
                        if cx - w // 2 < cursor[0] < cx + w // 2 and \
                                cy - h // 2 < cursor[1] < cy + h // 2:
                            rect.update(cursor, rectList)  # This will set dragging=True if valid
                            is_any_box_dragged = True
                            break
            else:  # Dragging pinch released
                for rect in rectList:
                    rect.dragging = False

    else:  # No hand detected
        for rect in rectList:
            rect.dragging = False
        drawing_box = False  # Reset drawing state if hand leaves
        start_point = None
        end_point = None

    # --- Drawing Existing Rectangles ---
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        draw_color = colorR if not rect.dragging else (0, 255, 0)  # Green when dragging
        cv2.rectangle(img, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), draw_color, cv2.FILLED)
        cvzone.cornerRect(img, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

    # --- Drawing the Live "Preview" Box ---
    if drawing_box and start_point and end_point:
        # Ensure points are ordered for cv2.rectangle (top-left, bottom-right)
        x1, y1 = start_point
        x2, y2 = end_point

        # Draw a temporary rectangle as you are defining its size
        cv2.rectangle(img, (min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2)), (0, 255, 255), 2)  # Yellow outline

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()