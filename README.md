# Face Recognition Based Attendance System



## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Algorithm Overview](#algorithm-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The Face Recognition Based Attendance System is a robust and efficient application designed to automate the attendance process using facial recognition technology. Leveraging the K-Nearest Neighbors (KNN) algorithm, this system provides an accurate and user-friendly method to track attendance, minimizing manual input and maximizing efficiency.

## Features
- **Automated Attendance**: Automatically marks attendance based on facial recognition.
- **Real-time Processing**: Uses a live video feed to detect faces in real time.
- **User-Friendly Interface**: Simple and intuitive interface for both teachers and students.
- **Customizable**: Easy to modify and extend according to specific requirements.
- **Data Management**: Store and manage attendance records efficiently.

## Technologies Used
- Python
- OpenCV
- scikit-learn
- Flask (for web framework)
- NumPy
- Pandas
- Matplotlib (for data visualization)

## Algorithm Overview
### K-Nearest Neighbors (KNN)
KNN is a simple yet effective algorithm used for classification tasks. In this project, KNN is utilized to classify the faces of users based on their features. The algorithm works as follows:
1. **Feature Extraction**: Faces are pre-processed and features are extracted using techniques like Histogram of Oriented Gradients (HOG).
2. **Training**: The KNN algorithm is trained with labeled data (images of users).
3. **Classification**: When a new face is detected, the algorithm compares it to the existing data and identifies the closest match based on distance metrics.

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/iamaindrik/face-attendance-system.git
   cd face-attendance-system
