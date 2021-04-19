


import cv2
import os
import numpy as np




videoCaptureObject = cv2.VideoCapture(0)
count=100

while(count!=0):
    ret,frame = videoCaptureObject.read()
    cv2.imwrite(r"C:\Users\SWAPNIL\Desktop\New folder\0\swap%04i.jpg"%count,frame)
    count=count-1
videoCaptureObject.release()
cv2.destroyAllWindows()


import time
  

def countdown(t):
    
    while t:
        mins, secs = divmod(t, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        print(timer, end="\r")
        time.sleep(1)
        t -= 1
      
   
  
  

t = input("Enter 5 sec.")
  

countdown(int(t))



videoCaptureObject = cv2.VideoCapture(0)
count=100

while(count!=0):
    ret,frame = videoCaptureObject.read()
    cv2.imwrite(r"C:\Users\SWAPNIL\Desktop\New folder\1\swap%04i.jpg"%count,frame)
    count=count-1
videoCaptureObject.release()
cv2.destroyAllWindows()




def faceDetection(test_img):
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)#convert color image to grayscale
    face_haar_cascade=cv2.CascadeClassifier(r"C:\Users\SWAPNIL\Desktop\haarcascade_frontalface_default.xml")#Load haar classifier
    faces=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.32,minNeighbors=5)#detectMultiScale returns rectangles

    return faces,gray_img


def labels_for_training_data(directory):
    faces=[]
    faceID=[]

    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping system file")#Skipping files that startwith .
                continue

            id=os.path.basename(path)#fetching subdirectory names
            img_path=os.path.join(path,filename)#fetching image path
            print("img_path:",img_path)
            print("id:",id)
            test_img=cv2.imread(img_path)
            if test_img is None:
                print("Image not loaded properly")
                continue
            faces_rect,gray_img=faceDetection(test_img)
            if len(faces_rect)!=1:
               continue 
            (x,y,w,h)=faces_rect[0]
            roi_gray=gray_img[y:y+w,x:x+h]
           
            faces.append(roi_gray)
            faceID.append(int(id))
    return faces,faceID



def train_classifier(faces,faceID):
    face_recognizer_LBPH=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer_Eigen=cv2.face.EigenFaceRecognizer_create()
    face_recognizer_Fisher=cv2.face.FisherFaceRecognizer_create()
    face_recognizer_LBPH.train(faces,np.array(faceID))
    
    face_recognizer_Eigen.train(faces,np.array(faceID))
    face_recognizer_Fisher.train(faces,np.array(faceID))
    return face_recognizer_LBPH,face_recognizer_Eigen,face_recognizer_Fisher


def draw_rect(test_img,face):
    (x,y,w,h)=face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5)


def put_text(test_img,text,x,y):
    cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,2,(255,0,0),4)




videoCaptureObject = cv2.VideoCapture(0)
ret,frame = videoCaptureObject.read()
cv2.imwrite(r"C:\Users\SWAPNIL\Desktop\Test.jpg",frame)
test_img=cv2.imread(r"C:\Users\SWAPNIL\Desktop\Test.jpg")#test_img path
faces_detected,gray_img= faceDetection(test_img)
print("faces_detected:",faces_detected)



faces,faceID=labels_for_training_data(r"C:\Users\SWAPNIL\Desktop\New folder")
face_recognizer_LBPH,face_recognizer_Eigen,face_recognizer_Fisher=train_classifier(faces,faceID)
face_recognizer_LBPH.write('trainingData.yml')
face_recognizer_Eigen.write('trainingData1.yml')
face_recognizer_Fisher.write('trainingData2.yml')




name={0:"Without",1:"With"}

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label_LBPH,confidence_LBPH=face_recognizer_LBPH.predict(roi_gray)#predicting the label of given image
    print("confidence_LBPH:",confidence_LBPH)
    print("label_LBPH:",label_LBPH)
    draw_rect(test_img,face)
    predicted_name=name[label_LBPH]
    if(confidence_LBPH>100):
        continue
    put_text(test_img,predicted_name,x,y)
    
    
    label_eigen,confidence_eigen=face_recognizer_Eigen.predict(roi_gray)
    print("confidence_eigen:",confidence_eigen)
    print("label_eigen:",label_eigen)
    
    
    
    label_fisher,confidence_fisher=face_recognizer_Fisher.predict(roi_gray)
    print("confidence_fisher:",confidence_fisher)
    print("label_fisher:",label_fisher)
    
    

resized_img=cv2.resize(test_img,(1000,1000))
cv2.imshow("face dtecetion tutorial",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows

videoCaptureObject.release()
















