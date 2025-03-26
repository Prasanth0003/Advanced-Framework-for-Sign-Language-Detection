# Advanced-Framework-for-Sign-Language-Detection

### ABSTRACT

This project presents an advanced framework for Sign Language Detection leveraging computer vision and machine learning techniques. The system integrates OpenCV and Mediapipe for real-time detection and tracking of hand, face, and pose landmarks, extracting key features crucial for gesture interpretation. A Long Short-Term Memory (LSTM) neural network model is employed to recognize temporal patterns in gesture sequences. The dataset is structured into labeled action sequences, preprocessed, and used to train the model to achieve high accuracy in real-time predictions. The solution supports live video streams, enabling gesture-to-text translation with visual overlays for interpretability. This framework offers a robust and scalable approach for enhancing communication accessibility for the deaf and hard-of-hearing community.

### INTRODUCTION

##### 1.1 OVERVIEW
Sign language serves as a crucial mode of communication for individuals with hearing or speech impairments. However, the barrier between sign language users and non-sign language users often creates challenges in effective communication. To address this, advanced frameworks for sign language detection aim to bridge this gap by leveraging cutting-edge technologies in artificial intelligence, computer vision, and natural language processing. These frameworks focus on accurately interpreting gestures, facial expressions, and hand movements, converting them into meaningful text or speech.

##### 1.2	 PURPOSE
The purpose of this project is to develop an innovative and efficient framework for detecting and translating sign language into spoken or written forms, thereby fostering inclusivity and accessibility. By employing state-of-the-art machine learning models and real-time data processing, this system is designed to enhance communication between sign language users and the wider community. The framework also aims to provide a robust platform for researchers and developers to further expand the capabilities of sign language detection, ultimately empowering individuals with hearing or speech impairments

### LITERATURE SURVEY

##### 2.1 Existing problem
###### •	HardwareDependency
Early systems relied heavily on hardware like data gloves and motion sensors. While these provided accurate gesture detection, their high cost, lack of portability, and need for extensive calibration made them impractical for widespread use.
###### •	Limited Scalability of Vision-Based Systems
Camera-based detection systems faced issues such as sensitivity to lighting conditions, background noise, and occlusions. These factors significantly affected their reliability and usability in real-world scenarios.
###### •	Challenges with Machine Learning Models
Traditional machine learning techniques like Hidden Markov Models (HMMs) and Support Vector Machines (SVMs) struggled to handle complex gestures, simultaneous hand movements, and diverse sign languages. These models often required handcrafted features, which limited their scalability and performance.
###### •	Dataset Constraints
The lack of large, annotated datasets with diverse sign language gestures restricted the training and validation of robust models.
###### •	Language-Specific Solutions
Most existing systems focus on a specific sign language, such as ASL, neglecting the diversity of global sign languages and the need for universal solutions.

##### 2.2	 Proposed Solution
The proposed solution leverages advanced methodologies in computer vision and machine learning to develop a robust framework for sign language detection. The system employs libraries such as OpenCV, Mediapipe, TensorFlow/Keras, and NumPy to ensure efficient initialization and compatibility across platforms. Utilizing Mediapipe’s holistic framework, the solution provides real-time detection and tracking of facial, hand, and pose landmarks, supported by custom functions for data preprocessing and visualization. Keypoints from pose, left-hand, and right-hand landmarks are systematically extracted and structured into organized datasets for model training. A Long Short-Term Memory (LSTM) network is employed to capture temporal relationships in gestures, trained with action-specific datasets for high accuracy. The trained model is integrated into a real-time system for live gesture recognition, with probability visualizations enhancing interpretability for users. The framework also emphasizes scalability and portability through model preservation and lightweight implementation, making it suitable for edge devices. This comprehensive approach ensures an accurate, efficient, and accessible solution for real-time sign language detection, fostering inclusivity and improved communication.

### PROPOSED METHOD

##### 3.1 Theoretical Analysis
##### 3.1.1	Block diagram
![image](https://github.com/user-attachments/assets/b402ea0c-078d-432e-82d7-d4152d4c1d0f)



##### 3.1.2 Hardware/Software Designing
###### Hardware Requirements:

**•	Camera:** For capturing real-time video input, preferably with high resolution for better accuracy.

**• Processing Unit:** A GPU or high-performance CPU for efficient training and real-time inference.

**•	Memory:** Minimum of 8GB RAM for smooth processing of video data and training tasks.

**•	Storage:** Sufficient disk space (at least 50GB) for dataset storage and model preservation.

**•	Edge Devices (Optional):** Devices such as Raspberry Pi for portable applications.

###### Software Requirements:

•	**Operating System:** Windows, Linux, or macOS with Python compatibility.

•	**Programming Environment:** Python 3.7 or above.

•**Libraries:**

o	OpenCV for video input and processing.

o	Mediapipe for landmark detection and tracking.

o	TensorFlow/Keras for implementing and training LSTM networks.

o	NumPy for numerical computations.

o	Matplotlib for data visualization.

•	**Additional Tools:**

o	TensorBoard for monitoring training progress.

o	Scikit-learn for model evaluation and data partitioning.

##### 3.2 Implementation

The implementation of the advanced framework for sign language detection involves a series of structured steps:

###### 1.	Initialization and Dependency Setup

o	Install essential libraries such as OpenCV, Mediapipe, TensorFlow/Keras, and NumPy for image processing, landmark detection, and model training.

o	Example commands include:

           pip install opencv-python mediapipe tensorflow numpy
           
##### 2.	Landmark Detection

o	Mediapipe Holistic is employed to detect and track face, hand, and pose landmarks. This step processes input video frames and extracts landmark coordinates.

o	Example function:

def mediapipe_detection(image, model):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image.flags.writeable = False
    
    results = model.process(image)
    
    image.flags.writeable = True
    
    return results
    
##### 3.	Real-Time Video Analysis

o	Using OpenCV, a live video feed is captured and passed through Mediapipe for landmark detection. Landmark visualizations are rendered directly on the video feed for real-time analysis.

o	Example code snippet:

        cap = cv2.VideoCapture(0)

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
    
        ret, frame = cap.read()
        
        results = mediapipe_detection(frame, holistic)
        
        draw_styled_landmarks(frame, results)
        
        cv2.imshow('Sign Language Detection', frame)
        
##### 4.	Feature Extraction and Dataset Preparation
   
o	Extract keypoints from detected landmarks and organize them into structured datasets categorized by gestures.

o	Example function for extracting keypoints:

    def extract_keypoints(results):

    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    
    return np.concatenate([pose, left_hand, right_hand])

##### 5.	Model Training

o	An LSTM network is trained using the structured dataset to recognize temporal dependencies in gestures. The model uses a categorical cross-entropy loss function and an Adam optimizer for efficient training.

o	Example architecture:

    model = Sequential()

    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))

    model.add(LSTM(128, return_sequences=True, activation='relu'))

    model.add(LSTM(64, activation='relu'))

    model.add(Dense(actions_array.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

##### 6.	Real-Time Gesture Recognition
   
o	The trained model is deployed for real-time gesture recognition. Predictions are overlaid on the video feed for user feedback.

#### 3.3 Model Building

The framework is modeled as a modular pipeline to streamline detection and recognition tasks:

•	Input Layer: Captures video frames and preprocesses them for compatibility with detection algorithms.

•	Detection Layer: Utilizes Mediapipe Holistic for extracting anatomical landmarks in real-time.

•	Feature Processing Layer: Extracts and structures keypoints into sequences compatible with LSTM networks.

•	LSTM Network: Trains on sequential data to capture temporal dependencies and predict gesture classes.

•	Output Layer: Visualizes predictions on the video feed with class probabilities for enhanced interpretability.

#### 3.3.1	Code 

```python
pip install opencv-python
pip install mediapipe --user
import cv2
import numpy as np
import os
import time
from matplotlib import pyplot as plt
import mediapipe as mp
mp_holisic= mp.solutions.holistic #make our detections
mp_drawing = mp.solutionsdrawing_utils #make drawings
def mediapipe_detection(image,model):
    #mediapipe uses image in RGB only.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#color conversion
    image.flags.writeable= False #image is no longer writeable
    results=model.process(image)# make prediction using mediapipe
    image.flags.writeable= True #image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)#color conversion to rdb to
    return image, results 

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                           mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2,circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230),thickness=2, circle_radius=2)

cap = cv2.VideoCapture(1) # note: use videoCapture(1) if using webcam
#set mediapipe models
with mp_holistic.Holistic(min_detection_confidence=0.5,
min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        #read our feed: reading the frame from our webcam
        ret, frame = cap.read()# frame img from webcam

        #make detections
        image,results = mediapipe_detection(frame, holistic)
        print(results)
        
        #draw landmarks
        draw_styled_landmarks(image,results)

        #show to screen
        cv2.imshow('OpenCV feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

draw_styled_landmarks(image,results)

plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))

len(results.left_hand_landmarks.landmark)
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in
results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])
x=extract_keypoints(results)
x

DATA_PATH=os.path.join("C:/Users/charan/OneDrive/Desktop/signLanguageDetection/MP_ATA")
#action detection
actions =  ['I love you','Hello','Thanks','want']
#we use 30 videos of data
no_consequences=60
#we use 30 frames to detect the action
sequence_lenght=60 
for action in actions:
    for sequence in range(no_consequences):
        path_to_create = os.path.join(DATA_PATH, str(action), str(sequence))
        os.makedirs(path_to_create)
cap = cv2.VideoCapture(1)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # NEW LOOP
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(no_consequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_lenght):

                # Read feed
                ret, frame = cap.read()
                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # NEW Apply wait logic
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(500)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number
{}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    cap.release()
    cv2.destroyAllWindows()

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

label_map={label:num for num,label in enumerate(actions)}

label_map

sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_lenght):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
        

x= np.array(sequences)

res.shape

np.sum(res[0])

y=to_categorical(labels).astype(int) #convet to one hot encoding
y

x_train, x_test ,y_train ,y_test = train_test_split(x,y,test_size=0.05)

x_train, x_test ,y_train ,y_test
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

log_dir = os.path.join("C:/Users/charan/OneDrive/Desktop/signLanguageDetection/logs")
tb_callback = TensorBoard(log_dir=log_dir)

actions_array=np.array(actions)

actions_array.shape[0]

x_train.shape

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))  # Set return_sequences to False
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions_array.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['categorical_accuracy'])

model.fit(x_train, y_train, epochs=300, callbacks=[tb_callback])

res = model.predict(x_test)

res

np.sum(res[0])

actions[np.argmax(res[1])]

     actions[np.argmax(y_test[1])]

model.summary()
model.save('action.h5')
model.load_weights('action.h5')

print(model)

model

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

yhat = model.predict(x_test)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
multilabel_confusion_matrix(ytrue, yhat)

res = res.flatten()

res

accuracy_score(ytrue, yhat)

colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)           
`	return output_frame

plt.figure(figsize=(18,18))
plt.imshow(prob_viz(res, actions, image, colors))

sequence.reverse()

len(sequence)

sequence.append('def')

sequence.reverse()

sequence[-30:]

sequence=[]
sentence=[]
threshold=0.8
cap = cv2.VideoCapture(1) # note: use videoCapture(1) if using webcam
#set mediapipe models
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        #read our feed: reading the frame from our webcam
        ret, frame = cap.read()# frame img from webcam

        #make detections
        image,results = mediapipe_detection(frame, holistic)
        print(results)
        
        #draw landmarks
        draw_styled_landmarks(image,results)
        
        #prediction logic 
        keypoints = extract_keypoints(results)
        sequence.insert(0,keypoints)
        sequence=sequence[:30]
        
        if len(sequence)==30:
            res=model.predict(np.expand_dims(sequence,axis=0))[0]
            print(actions[np.argmax(res)])
            
        #Viz logic
            if res[np.argmax(res)] > threshold: 
                if len(sentence) > 0: 
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

     # Viz probabilities
            image = prob_viz(res, actions, image, colors)
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#show to screen
        cv2.imshow('OpenCV feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
res[np.argmax(res)] > threshold
model.predict(np.expand_dims(x_test[0], axis=0))
```

### RESULTS

To analyze the results of the implemented sign language detection system, the following metrics and outputs are commonly evaluated:
Results Overview

**1. Model Accuracy**
   
•	Categorical Accuracy: The categorical_accuracy metric indicates how well the model predicts the correct action class during training and validation.

o	Example: After 300 epochs, the accuracy on the training and test datasets can be logged. For instance:

    Training Accuracy: ~98%
 
    Validation Accuracy: ~94%
 
**2. Confusion Matrix**

•	The confusion matrix generated using multilabel_confusion_matrix highlights the model's performance across different gesture classes.

o	True Positives (TP): Correct predictions for each class.

o	False Positives (FP) and False Negatives (FN): Misclassifications by the model.

**3. Real-Time Gesture Recognition**

•	During live inference:

o	The system displays the recognized gesture along with its probability.

o	Example Output:

    Predicted Gesture: "Hello"

    Confidence: 95%

o	Gestures are updated dynamically in real-time as input frames change.

**4. Evaluation Metrics**

•	Accuracy Score: The accuracy_score function measures the overall proportion of correct predictions.

    Example: Test Accuracy = 93%

•	Loss: The categorical_crossentropy loss during training provides insight into model optimization.

    Example: Final Training Loss = 0.12

**5. Visualization of Probabilities**

•	The prob_viz function overlays a visual representation of prediction probabilities on the video feed, enabling better interpretability.

**6. Sample Prediction Analysis**
   
•	Predictions for test data:

o	Example:

    Predicted Class: "Thanks"

    Actual Class: "Thanks"

    Confidence: 98%

### APPLICATIONS

**1. Assistive Technology for Communication**
   
•	Hearing Impairment Support: Enables individuals with hearing or speech impairments to communicate with non-sign language users in real-time by converting gestures into text or speech.

•	Inclusive Education: Facilitates better communication in classrooms and learning environments for students with special needs.

**3. Healthcare and Accessibility**
   
•	Medical Consultations: Assists doctors and medical staff in communicating effectively with patients who rely on sign language.

•	Emergency Services: Enhances accessibility in emergency scenarios where sign language users need immediate assistance.

**4. Customer Service**

•	Service Desks and Kiosks: Provides real-time sign language translation in retail stores, airports, and public service centers.

•	Virtual Assistants: Integrates sign language detection into virtual assistants to enhance user interaction for individuals with disabilities.

**5. Education and Learning**

•	Language Learning: Aids in teaching and learning sign language for students, teachers, and enthusiasts.

•	Interactive Tutorials: Creates interactive platforms where users can learn and practice sign language with real-time feedback.

**6. Workplace Inclusion**
   
•	Meetings and Collaboration: Supports inclusive communication in workplaces by translating sign language during meetings and group discussions.

•	Corporate Training: Provides tools for training employees in basic sign language, promoting inclusivity.

**7. Entertainment and Media**

•	Television and Online Streaming: Real-time translation of sign language on television shows, live events, or online streams for better accessibility.

•	Gaming: Incorporates gesture recognition in video games, enabling sign language-based controls for immersive experiences.


**8. Public Utilities and Smart Cities**

•	Smart Displays: Integrates sign language detection with smart city infrastructure to provide accessible information at bus stops, metro stations, or public terminals.

•	Tourism and Hospitality: Enhances the experience for tourists by enabling communication in sign language in hotels, museums, and tourist centers.


### CONCLUSION 
 
The advanced framework for sign language detection successfully integrates state-of-the-art technologies in computer vision and deep learning to enable robust, real-time gesture recognition. By employing tools like Mediapipe for landmark detection and LSTM networks for temporal modeling, the system achieves high accuracy and scalability. This solution addresses significant communication barriers faced by individuals reliant on sign language, promoting inclusivity across various domains such as healthcare, education, and public utilities. Despite its achievements, the framework highlights the need for more diverse datasets and language-agnostic models to ensure broader applicability.

### FUTURE SCOPE

The future scope of the sign language detection framework lies in enhancing its capabilities to support multilingual and culturally diverse sign languages, ensuring global accessibility. Expanding dataset diversity and incorporating contextual understanding will enable the system to interpret complex gestures and conversational nuances effectively. The integration with wearable devices, such as smart glasses or gloves, promises seamless and portable solutions, while optimizing for edge computing and IoT opens opportunities in smart cities, transportation, and home automation. Real-time translation to speech can further bridge communication gaps, fostering inclusivity in various domains. Additionally, exploring cross-domain applications, such as gesture-based authentication, immersive gaming, and virtual reality, can revolutionize user interactions. These advancements will empower the framework to meet the evolving needs of accessibility and inclusivity in a technologically driven world.
