from ultralytics import YOLO
import cv2
import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import os

#load model
model = YOLO('/Users/vinayakkanojia/Desktop/Glove detection/trained models/best.pt')

# print(model.names)

def predict_frame(frame, confidence_thresh):
    result = model.predict(source=frame, show=False, conf=confidence_thresh)

    #extract results
    boxes = result[0].boxes.xyxy
    confs = result[0].boxes.conf
    class_ids = result[0].boxes.cls

    # Initialize counters
    glove = no_glove = 0

    for box, confidence_score, class_id in zip(boxes, confs, class_ids):
        x1, y1, x2, y2 = map(int, box)
        class_name = model.names[int(class_id)]
        label = f"{class_name} {confidence_score:.2f}"

        #count

        if class_name == 'glove':
            glove+=1

        elif class_name == 'no_glove':
            no_glove+=1
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (201, 120, 255), 3)

        # Put label above box
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)
        cv2.rectangle(frame, (x1, y1 - text_h - 4), (x1 + text_w, y1), (255, 255, 255), -1)
        cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

    pred_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return [pred_img, glove, no_glove]

#---------------------------------------------------------------------------------------------------------

st.title("YOLOv11-Based Glove Detection")
st.sidebar.header("Toggle settings", divider='rainbow')
con_thresh = st.sidebar.slider("Select the Confidence threshold value", min_value=0.0, max_value=1.0, step=0.01, value= 0.4)
img = st.file_uploader("Insert the image here. \n choose it from you local pc", type=['jpg','jpeg','png','mp4','avi'], accept_multiple_files=False)

if img is not None:
    print(img)
    if img.type.startswith("image"):
        pil_image = Image.open(img)

        # Convert PIL → NumPy (RGB)
        pic = np.array(pil_image)

        # If you want OpenCV's default BGR
        pic = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)

        pic = predict_frame(pic, con_thresh)

        st.image(pic[0], use_container_width=True)
        st.sidebar.write(f'count of hands with glove: {pic[1]}')
        st.sidebar.write(f'count of hands without glove: {pic[2]}')

    elif img.type.startswith("video"):
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(img.read())
        temp_file.close()


        vid = cv2.VideoCapture(temp_file.name)
        # st.write(f'{temp_file.name}') # testing what is the temp_file.name returning. it returns a path to the video whi
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)

        frame_num = 0

        stframe = st.empty() 
        glovecountbox = st.sidebar.empty()
        noglovecountbox = st.sidebar.empty()
        
        fps_skip = st.sidebar.slider("Skip Frames (Higher = Faster)", 1, 10, 1) # this is added to regulate fps.

        stop_button = st.sidebar.button("Stop Video")
        while True:
            flag, frame = vid.read()
            if flag==False:
                break

            if stop_button:
                break

            pred_frame = predict_frame(frame,con_thresh)
            stframe.image(pred_frame[0], use_container_width=True)

            frame_num +=1
            if frame_num % fps_skip != 0:  #skip the frame that don't match
                continue
            progress_bar.progress(frame_num/total_frames)

            glovecountbox.write(f'count of hands with glove: {pred_frame[1]}')
            noglovecountbox.write(f'count of hands without glove: {pred_frame[2]}')
        vid.release()
        os.remove(temp_file.name)


