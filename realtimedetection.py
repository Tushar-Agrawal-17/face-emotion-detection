import cv2
import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt

# ---------------- LOAD MODEL ----------------

json_file=open("emotiondetector.json","r")
model_json=json_file.read()
json_file.close()

model=model_from_json(model_json)
model.load_weights("emotiondetector.h5")

print("Model loaded successfully")

# ---------------- FACE CASCADE ----------------

face_cascade=cv2.CascadeClassifier(
    cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
)

# ---------------- LABELS ----------------

labels={
0:'angry',
1:'disgust',
2:'fear',
3:'happy',
4:'sad',
5:'surprise',
6:'neutral'
}

# ---------------- EMOTION COUNTER ----------------

emotion_count={
'angry':0,
'disgust':0,
'fear':0,
'happy':0,
'sad':0,
'surprise':0,
'neutral':0
}

# ---------------- FEATURE EXTRACTION ----------------

def extract_features(image):

    feature=np.array(image)
    feature=feature.reshape(1,48,48,1)

    return feature/255.0


# ---------------- WEBCAM ----------------

webcam=cv2.VideoCapture(0)

while True:

    ret,frame=webcam.read()

    if not ret:
        break

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces=face_cascade.detectMultiScale(gray,1.3,5)

    for i,(x,y,w,h) in enumerate(faces):

        face=gray[y:y+h,x:x+w]

        face=cv2.resize(face,(48,48))

        face=extract_features(face)

        pred=model.predict(face,verbose=0)

        emotion_index=pred.argmax()

        prediction_label=labels[emotion_index]

        confidence=pred[0][emotion_index]*100

        # update emotion count
        emotion_count[prediction_label]+=1

        # rectangle
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        text=f"Person {i+1} : {prediction_label} {confidence:.1f}%"

        cv2.putText(frame,
                    text,
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,0,255),
                    2)

    cv2.imshow("Emotion Detection",frame)

    # press q to exit
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break


webcam.release()
cv2.destroyAllWindows()

# ---------------- SESSION REPORT ----------------

print("\nEmotion Session Report")
print("----------------------")

for emotion,count in emotion_count.items():
    print(emotion,":",count)

# ---------------- GRAPH ----------------

plt.bar(emotion_count.keys(),emotion_count.values())

plt.title("Emotion Distribution")

plt.xlabel("Emotion")

plt.ylabel("Frequency")

plt.show()