
import dlib
import cv2
import numpy as np

def aspect(eye):
    x0=eye[0,0]
    y0=eye[0,1]
    x1=eye[1,0]
    y1=eye[1,1]
    x2=eye[2,0]
    y2=eye[2,1]
    x3=eye[3,0]
    y3=eye[3,1]
    x4=eye[4,0]
    y4=eye[4,1]
    x5=eye[5,0]
    y5=eye[5,1]
    
    xmid12=(x1+x2)/2
    ymid12=(y1+y2)/2
    xmid45=(x4+x5)/2
    ymid45=(y4+y5)/2
    
    p=(((xmid45-xmid12)**(2))+((ymid45-ymid12)**(2)))**(0.5)
    
    q=(((x3-x0)**(2))+((y3-y0)**(2)))**(0.5)
    

    ratio=p/q
    return ratio	

threshold=0.25
limit=15
ffd=dlib.get_frontal_face_detector()
sp=dlib.shape_predictor(".\shape_predictor_68_face_landmarks.dat")
cap=cv2.VideoCapture(0)
temp=0

def shape_to_np(shape, dtype="int"):
	
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	
	return coords

while True:
    ret, frame=cap.read()

    gs=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    j=ffd(gs,0)
    for i in j:
        shape=sp(gs,i)
        shape=shape_to_np(shape)
        print(shape)
        leftEye=shape[42:48]
        rightEye=shape[36:42]
        leftaspect=aspect(leftEye)
        rightaspect=aspect(rightEye)
        if leftaspect<threshold and rightaspect<threshold:
            temp+=1 
            
            if temp>=limit:
                cv2.putText(frame, "DANGER DANGEER DANGER", (185, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "DANGER DANGER DANGER", (185,430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)			
        else:
            temp=0
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1) & 0xFF
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()
