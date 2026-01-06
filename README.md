# AI_project_FlowSense
Vision-based AI system for airflow detection FlowSense project using a single yarn tell-tale and a standard webcam. This prototype uses Deep Learning (MobileNetV2) to classify yarn movement without expensive sensors.

Physical lab setup and measurements 
For the model to perform accurately, the physical environment must match the training conditions. Please follow measurements for the rig setup.

Camera and object placement
Camera Distance: 30 cm from the target object (ball/yarn center).
Reference Object (Ball) Dimensions:
Radius: 3.25 cm
Diameter ($D$): 6.5 cm

Object elevation:
Ball height from surface (measured from center): 18 cm.
Stick height from table surface: 35 cm.

Spacing:
Distance between Ball and Stick (from center): 18 cm.
Distance between hanging yarn strands: 5 - 7 cm.

Tech Stack 
Language: Python 3.x 
Vision: OpenCV (cv2) 
ML/DL: PyTorch (MobileNetV2 Transfer Learning) 
Data: Pandas, Matplotlib (for Dashboard)

Installation:
pip install torch torchvision opencv-python pandas matplotlib

Usage pipeline
Follow this order to go from raw video to real-time detection.

1. Data preparation
Record videos of specific airflow states (e.g., left_calm.mp4, stationary.mp4) and place them in src/data/raw_videos.

Run the extractor to crop, resize (224x224), and save frames:
python extract_frames.py

2. Dataset organization
Sort the extracted frames into training and validation sets based on folder names:
python preprocess.py

3. Model training
Train the MobileNetV2 model on your organized dataset:
python train_model.py

4. Real-time detection
Launch the webcam inference loop:
python real_time_detection.py
Controls: Press q to quit the video feed.
Feedback: The window displays the detected state (e.g., "LEFT_STRONG") and confidence score. Green bounding boxes indicate confidence > 60%.

5. Live dashboard
To visualize the frequency of states over the last 50 frames, run the dashboard in a separate terminal while detection is running:
python dashboard.py
This reads from src/data/wind_log.csv in real-time.


Technical details
Model architecture: MobileNetV2, modified with a custom linear classifier layer.
Input Resolution: 224 * 224 pixels.
Smoothing: A LabelSmoother class uses a rolling buffer to prevent prediction flickering by returning the most common label in the recent history.

Authors
Bilal and Arin - FlowSense project