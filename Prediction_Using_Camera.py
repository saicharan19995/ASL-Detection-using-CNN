from tensorflow.keras.models import load_model
import cv2,numpy as np

cap = cv2.VideoCapture(0)
x = 35
y = 80
w = h = 500
new_model=load_model('Models/Model_Epoch_5.h5')
values=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','space']
while(True):
    ret, frame = cap.read()
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    img = frame[x:x+w,y:y+h]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(200,200))
    x_test = np.array(img).reshape(-1,200,200,1)
    pre = new_model.predict_classes(x_test)
    cv2.putText(frame,str(values[pre[0]]),(10,500),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    # cv2.putText(frame,str(new_model.predict_proba(x_test)[0][pre[0]])*100,(10,450),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
