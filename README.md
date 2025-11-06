# Real-Time ASL Alphabet Classifier

A computer vision system that recognizes American Sign Language (ASL) alphabet gestures in real-time using hand landmark detection and machine learning.

## Project Overview

This project enables real-time recognition of ASL alphabet signs (A-Z) through webcam input. Using MediaPipe for hand tracking and a Random Forest classifier, the system can identify hand gestures with high accuracy, making it useful for learning ASL or building accessibility applications.

## Screenshots
<div align="center">
  
**Real-Time Recognition**

<img width="801" height="636" alt="image" src="https://github.com/user-attachments/assets/85570de8-ab3c-4e16-93b6-359b4d5fa68d" />

*Live webcam feed with hand tracking and letter prediction*

**Hand Landmark Detection**

<img width="304" height="284" alt="image" src="https://github.com/user-attachments/assets/1f781778-7ad4-4e44-9594-b0711cabc2b9" />

*MediaPipe detecting 21 hand keypoints*

**Multiple Gestures**

<img width="802" height="632" alt="image" src="https://github.com/user-attachments/assets/d3ee5c99-d72c-4afb-b9d9-e375e88cec80" />
<img width="799" height="633" alt="image" src="https://github.com/user-attachments/assets/68dfbd07-5504-4c4e-a45d-b0d3150259d8" />
<img width="795" height="632" alt="image" src="https://github.com/user-attachments/assets/9bdeaeb1-94b3-4be5-8aff-d6b2f321be04" />

*Examples of different ASL letters being recognized*
</div>


## Features

- **Real-Time Recognition**: Processes webcam feed at interactive frame rates
- **26-Class Classifier**: Recognizes all letters of the ASL alphabet (A-Z)
- **Position Invariant**: Hand position doesn't affect classification accuracy
- **Visual Feedback**: Displays hand landmarks, bounding boxes, and predictions on screen
- **Custom Dataset**: Includes complete pipeline for collecting and training on your own data
- **High Accuracy**: Random Forest classifier with optimized hyperparameters

## Tech Stack

- **MediaPipe**: Hand landmark detection (21 keypoints per hand)
- **OpenCV**: Real-time video processing and visualization
- **Scikit-learn**: Random Forest classifier
- **NumPy**: Numerical computations
- **Pickle**: Model serialization

## System Architecture

```
Webcam Input → MediaPipe Hand Detection → Feature Extraction → Random Forest Classifier → Prediction Display
```

**Feature Extraction**:
- 21 hand landmarks (x, y coordinates)
- Normalized relative to hand bounding box
- 42 total features (21 landmarks × 2 coordinates)

**Model Details**:
- Algorithm: Random Forest
- Estimators: 200 trees
- Max Depth: 20
- Min Samples Split: 5
- Train/Test Split: 80/20

## Results

- **Test Accuracy**: 99.90384615384616%
- **Classes**: 26 (A-Z alphabet)
- **Training Samples**: 5,200 images (200 per class)
- **Real-Time Performance**: Processes live video stream
- **Feature Dimensions**: 42 normalized landmark coordinates

## Getting Started

### Prerequisites

```
pip install opencv-python mediapipe scikit-learn numpy pickle-mixin
```

### Full Pipeline

#### 1. Collect Training Data

```
python collect_imgs.py
```

- Press 'Q' to start collecting for each class
- System captures 200 images per letter
- Perform the ASL gesture for each letter when prompted
- Data saved to `./data/` directory

#### 2. Create Feature Dataset

```
python create_dataset.py
```

- Processes all collected images
- Extracts hand landmarks using MediaPipe
- Normalizes coordinates for position invariance
- Saves features to `data.pickle`

#### 3. Train Classifier

```
python train_classifier.py
```

- Loads preprocessed features
- Trains Random Forest classifier
- Evaluates on test set (20% holdout)
- Saves trained model to `model.p`

#### 4. Run Real-Time Recognition

```
python inference_classifier.py
```

- Opens webcam feed
- Detects and tracks hand landmarks
- Displays predicted letter with bounding box
- Press 'Q' or 'ESC' to exit

## Project Structure

```
.
├── collect_imgs.py          # Data collection script
├── create_dataset.py        # Feature extraction pipeline
├── train_classifier.py      # Model training
├── inference_classifier.py  # Real-time inference
├── data/                    # Raw image data (26 folders)
│   ├── 0/                   # Class 0 (A)
│   ├── 1/                   # Class 1 (B)
│   └── ...
├── data.pickle              # Preprocessed features
├── model.p                  # Trained model
└── README.md
```

## How It Works

### 1. Hand Landmark Detection
MediaPipe detects 21 key points on the hand:
- Wrist
- Thumb (4 points)
- Index finger (4 points)
- Middle finger (4 points)
- Ring finger (4 points)
- Pinky (4 points)

### 2. Feature Engineering
```
# Raw coordinates
x, y = landmark.x, landmark.y

# Normalization (position invariance)
normalized_x = x - min(all_x_coords)
normalized_y = y - min(all_y_coords)
```

This ensures the classifier recognizes gestures regardless of where the hand appears in the frame.

### 3. Classification
Random Forest ensemble makes predictions based on normalized hand shape, not position or size.

## Sample Output

```
99.90384615384616% of samples were classified correctly

[Webcam displays with bounding box and predicted letter]
```

## Key Concepts Demonstrated

- End-to-end machine learning pipeline
- Computer vision with MediaPipe
- Real-time video processing
- Feature engineering and normalization
- Ensemble learning (Random Forest)
- Data collection and preprocessing
- Model persistence and deployment

## Use Cases

- **ASL Learning Tool**: Practice and get instant feedback
- **Accessibility**: Communication aid for deaf/hard-of-hearing individuals
- **Educational Demo**: Teaching ML and computer vision concepts
- **Base for Expansion**: Foundation for full sign language translation system

## Future Enhancements

- [ ] Add confidence scores to predictions
- [ ] Implement word/phrase recognition (not just letters)
- [ ] Support for numbers and common words
- [ ] Add prediction history/smoothing for stability
- [ ] Create web-based interface
- [ ] Multi-hand support for two-handed signs
- [ ] Add dataset augmentation (rotation, scaling)
- [ ] Implement CNN for potentially higher accuracy
- [ ] Add gesture recording and playback
- [ ] Support for other sign languages (BSL, etc.)

## Troubleshooting

**Camera not opening**:
- Check if another application is using the webcam
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

**Poor recognition accuracy**:
- Ensure good lighting conditions
- Keep hand clearly visible in frame
- Use plain background for better detection
- Collect more training data with varied hand positions

**Hand not detected**:
- Adjust `min_detection_confidence` in MediaPipe (default: 0.3)
- Ensure entire hand is visible in frame
- Check lighting conditions

## Tips for Best Results

- **Data Collection**: Vary hand position, distance, and angle for each gesture
- **Lighting**: Use consistent, good lighting during training and inference
- **Background**: Plain backgrounds work best for hand detection
- **Gestures**: Hold each sign clearly and consistently
- **Distance**: Keep hand at similar distance to camera as during training

## ASL Resources

- [ASL Alphabet Chart](https://www.startasl.com/american-sign-language-alphabet/)
- Practice with proper ASL form for best recognition results

## License

This project is open source and available for educational purposes.

## Acknowledgments

- MediaPipe by Google for hand tracking technology
- ASL community for gesture definitions
- MNIST-style approach adapted for gesture recognition

---

**Author**: Ishvir Singh Chopra  
**Date**: 05 Nov 2025  
**Contact**: ishvir.chopra@gmail.com  
**Demo**: Video coming soon!
