"""Run this to find arrow key codes on your system.
Press arrow keys one at a time, then press q to quit.
"""
import cv2
import numpy as np

img = np.zeros((100, 400, 3), dtype=np.uint8)
cv2.putText(img, "Press arrow keys, then q", (10, 50),
            cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 200), 1)
cv2.imshow("keytest", img)

while True:
    key = cv2.waitKey(0)
    print(f"key={key}  &0xFF={key & 0xFF}  &0xFFFF={key & 0xFFFF}")
    if key & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
