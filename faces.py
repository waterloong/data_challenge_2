

# detect face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
for i in range(Ytrain.shape[1]):
    faces = face_cascade.detectMultiScale(images[i])
    print faces