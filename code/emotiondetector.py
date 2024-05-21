import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model(r'code\TrainedModels\emotion_recognition_model.h5')

face_cascade = cv2.CascadeClassifier(r'code\haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

while True:
  ret, frame = cap.read()
  
  gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
  faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

  for (x, y, w, h) in faces:
    face_roi = gray_frame[y:y+h, x:x+w]
    
    def preprocess_image(img, target_size=(48, 48)):
      img = cv2.resize(img, target_size)
      img = img.astype('float32') / 255.0
      img = img.reshape(target_size[0], target_size[1], 1)
      return img

    preprocessed_face = preprocess_image(face_roi)
    preprocessed_face = np.expand_dims(preprocessed_face, axis=0)

    prediction = model.predict(preprocessed_face)
    
    predicted_emotion = emotion_labels[np.argmax(prediction[0])]

    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    print(predicted_emotion)

  cv2.imshow('Emotion Recognition', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
