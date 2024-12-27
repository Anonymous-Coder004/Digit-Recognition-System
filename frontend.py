import streamlit as st
import cv2 as cv
import numpy as np
import pickle as pkl
st.title("Digit Recognition System")
#model
pickle_in=open("model_trained.pkl","rb")
model=pkl.load(pickle_in)
#preprocessing
def preProcessing(img):
    img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img=cv.equalizeHist(img)#enhance the contrast so that text can be read easily
    img=img/255
    return img
# Initialize variables
cap = None
frame_placeholder = st.empty()
#buttons
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
# Toggle button to start/stop the webcam
toggle_button = st.button("Start/Stop Webcam")
if toggle_button:
    st.session_state.is_running = not st.session_state.is_running
if st.session_state.is_running:
    if cap is None:
        cap = cv.VideoCapture(0)
        cap.set(3, 640)  #width
        cap.set(4, 480)  #height
    while cap.isOpened() and st.session_state.is_running:
        ret, frame_orginal = cap.read()
        if not ret:
            st.write("Video capture has ended")
            break
        frame=np.asarray(frame_orginal)
        frame=cv.resize(frame,(32,32))
        frame=preProcessing(frame)
        frame=frame.reshape(1,32,32,1) #reshape the image to 4D
        predictions=model.predict(frame)
        classIndex=int(np.argmax(predictions,axis=1))
        probval=np.amax(predictions)
        if probval>0.65:
            cv.putText(frame_orginal,"NUMBER IS:"+str(classIndex),(50,50),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
        else:
            cv.putText(frame_orginal,"NUMBER NOT FOUND",(50,50),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
        frame_orginal = cv.cvtColor(frame_orginal, cv.COLOR_BGR2RGB)
        frame_placeholder.image(frame_orginal, channels="RGB")
        if cv.waitKey(2) & 0xFF == ord('q'):
            st.session_state.is_running = False
            break
if cap is not None and not st.session_state.is_running:
    cap.release()
    cv.destroyAllWindows()
    cap = None
