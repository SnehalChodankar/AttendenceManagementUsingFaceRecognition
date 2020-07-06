import cv2, os

imageDir =  'C:\\Users\\sneha\\Desktop\\AttendenceProject\\database\\Pooja'

cam = cv2.VideoCapture(0)
cv2.namedWindow("test")

img_counter = 0

while img_counter<=10:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)
    #label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    else:
        # SPACE pressed
        img_name = "{}.png".format(img_counter)
        cv2.imwrite(os.path.join(imageDir, img_name), frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()
cv2.destroyAllWindows()
