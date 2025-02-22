'''import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import imutils
import json
import tempfile
import os

def mean_squared_loss(x1, x2):
    diff = x1 - x2
    a, b, c, d, e = diff.shape
    n_samples = a * b * c * d * e
    sq_diff = diff ** 2
    total = sq_diff.sum()
    distance = np.sqrt(total)
    mean_distance = distance / n_samples

    return mean_distance

# Load the Keras model
model = load_model("model\\saved_model.keras")

# Function to perform anomaly detection on video frames and save abnormal frames
def detect_anomalies(video_file_path):
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        st.error("Error opening video file.")
        return

    frame_count = 0
    im_frames = []
    abnormal_frames = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        image = imutils.resize(frame, width=700, height=600)
        frame = cv2.resize(frame, (227, 227), interpolation=cv2.INTER_AREA)
        gray = 0.2989 * frame[:,:,0] + 0.5870 * frame[:,:,1] + 0.1140 * frame[:,:,2]
        gray = (gray - gray.mean()) / gray.std()
        gray = np.clip(gray, 0, 1)
        im_frames.append(gray)

        if frame_count % 10 == 0:
            im_frames = np.array(im_frames)
            im_frames.resize(227, 227, 10)
            im_frames = np.expand_dims(im_frames, axis=0)
            im_frames = np.expand_dims(im_frames, axis=4)

            output = model.predict(im_frames)

            loss = mean_squared_loss(im_frames, output)

            if loss > 0.00038:
                st.error('🚨 Abnormal Event Detected 🚨')
                st.image(image, caption="", channels="BGR")
                abnormal_frames.append(frame_count)

            im_frames = []

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save abnormal frames indices to JSON
    with open('abnormal_frames.json', 'w') as json_file:
        json.dump(abnormal_frames, json_file)

# Streamlit code for file upload and anomaly detection
st.markdown("<h1 style='text-align: center; color: #006699;'>DeepEYE Anomaly Surveillance 👁️</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    detect_anomalies(tmp_file_path)

    os.unlink(tmp_file_path)'''
'''import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import imutils
import json
import tempfile
import os

# Function to calculate Mean Squared Loss
def mean_squared_loss(x1, x2):
    diff = x1 - x2
    a, b, c, d, e = diff.shape
    n_samples = a * b * c * d * e
    sq_diff = diff ** 2
    total = sq_diff.sum()
    distance = np.sqrt(total)
    mean_distance = distance / n_samples
    return mean_distance

# Load the trained model
model = load_model("model/saved_model.keras")

# Function to perform anomaly detection on video frames
def detect_anomalies(video_file_path):
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        st.error("Error opening video file.")
        return

    frame_count = 0
    im_frames = []
    abnormal_frames = []
    anomaly_detected = False  # Flag to track if any anomaly is found

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        image = imutils.resize(frame, width=700, height=600)
        frame = cv2.resize(frame, (227, 227), interpolation=cv2.INTER_AREA)
        
        # Convert frame to grayscale and normalize
        gray = 0.2989 * frame[:,:,0] + 0.5870 * frame[:,:,1] + 0.1140 * frame[:,:,2]
        gray = (gray - gray.mean()) / gray.std()
        gray = np.clip(gray, 0, 1)
        im_frames.append(gray)

        if frame_count % 10 == 0:  # Process every 10 frames
            im_frames = np.array(im_frames)
            im_frames.resize(227, 227, 10)
            im_frames = np.expand_dims(im_frames, axis=0)
            im_frames = np.expand_dims(im_frames, axis=4)

            output = model.predict(im_frames)
            loss = mean_squared_loss(im_frames, output)

            if loss > 0.00038:
                st.error('🚨 Abnormal Event Detected 🚨', icon="🚨")
                st.image(image, caption="Abnormal Event Frame", channels="BGR")
                
                abnormal_frames.append(frame_count)
                anomaly_detected = True  # Set flag to True

            im_frames = []  # Reset frame buffer

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save abnormal frame indices to JSON
    with open('abnormal_frames.json', 'w') as json_file:
        json.dump(abnormal_frames, json_file)

    # 🔥 Play alarm sound only once if an anomaly was detected
    if anomaly_detected:
        st.audio("C:/Users/ajayk/OneDrive/major phase-2/DeepEYE-Video-Surveillance-with-Anomaly-Detection-main/alert-33762.mp3", autoplay=True)

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: #006699;'>DeepEYE Anomaly Surveillance 👁️</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())'''
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import imutils
import json
import tempfile
import os

def mean_squared_loss(x1, x2):
    diff = x1 - x2
    a, b, c, d, e = diff.shape
    n_samples = a * b * c * d * e
    sq_diff = diff ** 2
    total = sq_diff.sum()
    distance = np.sqrt(total)
    mean_distance = distance / n_samples
    return mean_distance

# Load the trained model
model = load_model("model/saved_model.keras")

# Function to perform anomaly detection on video frames
def detect_anomalies(video_file_path, threshold):
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        st.error("Error opening video file.")
        return

    frame_count = 0
    im_frames = []
    abnormal_frames = []
    anomaly_detected = False  # Flag to track if any anomaly is detected

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Resize for display
        image = imutils.resize(frame, width=700, height=600)
        # Resize frame for model input
        frame_resized = cv2.resize(frame, (227, 227), interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale and normalize
        gray = 0.2989 * frame_resized[:, :, 0] + 0.5870 * frame_resized[:, :, 1] + 0.1140 * frame_resized[:, :, 2]
        gray = (gray - gray.mean()) / gray.std()
        gray = np.clip(gray, 0, 1)
        im_frames.append(gray)

        # Process every 10 frames
        if frame_count % 10 == 0:
            # Prepare the frames in the expected shape: (1, 227, 227, 10, 1)
            frames_array = np.array(im_frames)
            frames_array = np.resize(frames_array, (227, 227, 10))
            frames_array = np.expand_dims(frames_array, axis=0)
            frames_array = np.expand_dims(frames_array, axis=4)

            output = model.predict(frames_array)
            loss = mean_squared_loss(frames_array, output)

            # Log the loss value so you can adjust the threshold if needed
            st.write(f"Processed Frame {frame_count} - Loss: {loss:.5f}")

            # Check if the loss exceeds the threshold
            if loss > threshold:
                st.error('🚨 Abnormal Event Detected 🚨', icon="🚨")
                st.image(image, caption="Abnormal Event Frame", channels="BGR")
                abnormal_frames.append(frame_count)
                anomaly_detected = True

            # Reset the frame buffer for the next block of frames
            im_frames = []

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save abnormal frame indices to a JSON file
    with open('abnormal_frames.json', 'w') as json_file:
        json.dump(abnormal_frames, json_file)

    # Play the alarm sound once if any anomaly was detected
    if anomaly_detected:
        st.audio("C:/Users/ajayk/OneDrive/major phase-2/DeepEYE-Video-Surveillance-with-Anomaly-Detection-main/alert-33762.mp3", autoplay=True)

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: #006699;'>DeepEYE Anomaly Surveillance 👁️</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

# Sidebar: Adjustable threshold for anomaly detection
threshold = st.sidebar.number_input(
    "Anomaly Threshold",
    value=0.00038,
    step=0.00001,
    format="%.5f",
    help="Lowering this value will trigger the alarm more easily. Increase it to reduce false alarms."
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    detect_anomalies(tmp_file_path, threshold)
    os.unlink(tmp_file_path)

        tmp_file_path = tmp_file.name

    detect_anomalies(tmp_file_path)  # Run anomaly detection
    os.unlink(tmp_file_path)  # Delete temp file after processing
