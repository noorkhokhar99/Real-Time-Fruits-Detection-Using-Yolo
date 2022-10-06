import cv2
import argparse
import numpy as np

argparser = argparse.ArgumentParser(description='Simple implementation of Yolov3 algorithm in Python, using custom Dataset.')

argparser.add_argument('--img', type=str)
argparser.add_argument('--video', type=str)
argparser.add_argument('--out', type=str)

args = argparser.parse_args()

confidence, threshold = 0.5, 0.3 

# Load the Custom class labels, Weights and Config files
# Then create the DNN model
labelPath = './obj.names'
labels = open(labelPath).read().strip().split('\n')
weightsPath = './yolov3-tiny.weights'
configPath = './yolov3-tiny.cfg'
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Get all layer names
# Then get all [yolo] layers
layer_names = net.getLayerNames()
yolo_layers = []
for i in net.getUnconnectedOutLayers():
    yolo_layers.append(layer_names[i[0] - 1])

def draw_bb(img, boxes, confidences, classids, idxs, labels):
    # If detection exists
    if len(idxs):
        for i in idxs.flatten():
            # Get BB coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 10)
            text = "{}:{:2.5f}".format(labels[classids[i]], confidences[i])
            cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 70, 0), 3)

    return img

    # init lists of detected boxes, confidences, class IDs
    boxes, confidences, class_ids = [], [], []

def predict(net, layer_names, height, width, img, labels):

    # Construct a blob from input
    # Then perform a forward pass of the yolo OD
    # Then get BB with associated probabilities
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(layer_names)

    # init lists of detected boxes, confidences, class IDs
    boxes, confidences, class_ids = [], [], []

    # loop over each of the layer outputs
    for out in layerOutputs:
        # loop over each of the detections
        for detection in out:
            # extract the class ID and confidence of the current OD
            scores = detection[5:]
            class_id = np.argmax(scores)
            detect_confidence = scores[class_id]

            # filter out a weal predictions by ensuring the detected
            # probability is greater than minimum probability

            if detect_confidence > confidence:
                # scale the BB coordinates back relative to
                # the size of the image.
                # YOLO returns the center (x,y) - coordinates of BB
                # followed by the boxes weight and height
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, box_width, box_height = box.astype('int')

                # use the center (x, y) - coordinates to derive
                # the top and left corner of the BB
                x = int(centerX - (box_width / 2))
                y = int(centerY - (box_height / 2))

                # update list of BB coordinates, confidences, and class IDs
                boxes.append([x, y, int(box_width), int(box_height)])
                confidences.append(float(detect_confidence))
                class_ids.append(class_id)

    # Suppress overlapping boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

    img = draw_bb(img, boxes, confidences, class_ids, idxs, labels)

    return img, boxes, confidences, class_ids, idxs

if args.img:
    img = cv2.imread(args.img)
    height, width = img.shape[:2]
    img, _, _, _, _ = predict(net, yolo_layers, height, width, img, labels)
    
    if args.out:
        cv2.imwrite(args.out, img)
    else:
        img = cv2.resize(img, (800, 800))
        cv2.imshow('Prediction', img)
        cv2.waitKey(0)

elif args.video:
    cap = cv2.VideoCapture(args.video)
    height, width = None, None
    writer = None

    while True:
        grabbed, frame = cap.read()
        print(grabbed, frame)
        if not grabbed:
            break

        if width is None or height is None:
            height, width = frame.shape[:2]

        frame, _, _, _, _ = predict(net, yolo_layers, height, width, frame, labels)
        
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args.out, fourcc, 60, (frame.shape[1], frame.shape[0]), True)
        
        writer.write(frame)

    writer.release()
    cap.release()