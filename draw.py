import cv2

labels = ["No Mask", "Mask"]
labelcolor = [(50, 50, 255), (50, 255, 25)]
font = cv2.FONT_HERSHEY_TRIPLEX


def drawing_on_frame(frame, xcor, ycor, width, height, result):
    # draw face frame
    cv2.rectangle(frame, (xcor, ycor), (xcor + width, ycor + height), labelcolor[result], thickness=2)

    # center text according to the face frame
    textSize = cv2.getTextSize(labels[result], font, 1, 2)[0]
    textX = xcor + width // 2 - textSize[0] // 2

    # draw prediction label
    cv2.putText(frame, labels[result], (textX, ycor - 20), font, 1, labelcolor[result], 2)
