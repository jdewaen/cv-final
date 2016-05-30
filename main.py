import cv2
from importer import import_image, PROCESSED_RADIO_AMOUNT, TOTAL_RADIO_AMOUNT, TEETH_AMOUNT

for image in range(1, TOTAL_RADIO_AMOUNT + 1):
    image_data = import_image(image)
    radiograph = image_data['radiograph']
    if image_data['full_data']:
        for tooth in image_data['original_landmarks']:
            for landmark in tooth:
                cv2.circle(radiograph, landmark, 2, (0, 0, 255), 2)
    cv2.imshow(str(image), cv2.resize(radiograph, (0, 0), fx=0.5, fy=0.5))
    cv2.waitKey()
    cv2.destroyWindow(str(image))
