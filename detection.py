
import cv2


# accuracy configuration
people_accuracy = 0.25
global_accuracy = 0.55

nms_accuracy = 0.45

# configure input video / here from webcam
webcam = cv2.VideoCapture(0)
webcam.set(3,1280)
webcam.set(4,720)
webcam.set(10,70)


# load some pre trained data on face frontal from opencv

trained_face_data = cv2.CascadeClassifier('ressource/haarcascade_frontalface_default.xml')

item_names= []
item_file = 'ressource/coco.names'
with open(item_file,'rt') as f:
    item_names = f.read().rstrip('\n').split('\n')

configPath = 'ressource/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'ressource/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success,img = webcam.read()
    classIds, confs, bbox = net.detect(img,confThreshold=global_accuracy, nmsThreshold=nms_accuracy)
    greyscale_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('greyscale test: ', greyscale_frame)
    #print('classIds: ', len(classIds), 'item\n', classIds,'\n\nbbox: ', len(bbox), 'item\n', bbox, '\n\nconfs: ', len(confs), 'item\n', confs)
    
    for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
        if classId == 1: # classId 1 = person
            if confidence > people_accuracy:
                # draw rectangle around
                print('img: ', len(img), '\n',img,'\n\nbox: ', len(box), '\n', box)
                cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                cv2.putText(img,item_names[classId-1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            	# detection face
                # if face is detected draw rectangle around face
                # else print some color around 
                #print('detect face now !!!')
                #print('confs: ', confs, '\n\n')
            	#print(box)
                face_coordinate = trained_face_data.detectMultiScale(greyscale_frame)
            	# draw rectangle around faces
                for (x, y, w, h) in face_coordinate:
                    cv2.rectangle(img, (x, y-50), (x+w+30, y+h+40), (0, 250, 0), 4)
            else:
                pass
   
    cv2.imshow("Output",img)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
    	break
    	webcam.release()
