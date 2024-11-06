# Face Recognition Based Attendance System

<p align="center"><img src="https://github.com/pande17827/Face_Recognition_Attendance_System/blob/main/face-recognition.jpg"></p>

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Algorithm Overview](#algorithm-overview)

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


    <h2>Setup and Running Instructions</h2>
    <h3>Step 1: Clone the Repository</h3>
    <p>Open your terminal or command prompt and run the following command:</p>
    <pre><code>git clone https://github.com/iamaindrik/face-attendance-system.git</code></pre>
    <h3>Step 2: Navigate to the Project Directory</h3>
    <pre><code>cd face-attendance-system</code></pre>
    <h3>Step 3: Set Up a Virtual Environment (Optional but Recommended)</h3>
    <p>Install <code>virtualenv</code> if you haven't:</p>
    <pre><code>pip install virtualenv</code></pre>
    <p>Create a virtual environment:</p>
    <pre><code>virtualenv venv</code></pre>
    <p>Activate the virtual environment:</p>
    <ul>
        <li>On <strong>Windows</strong>: <pre><code>venv\Scripts\activate</code></pre></li>
        <li>On <strong>macOS/Linux</strong>: <pre><code>source venv/bin/activate</code></pre></li>
    </ul>
    <h3>Step 4: Install Dependencies</h3>
    <p>Run the following command to install all required packages:</p>
    <pre><code>pip install -r requirements.txt</code></pre>
    <h3>Step 5: Run the Application</h3>
    <p>Use the following command to start the Flask application:</p>
    <pre><code>python app.py</code></pre>
    <h3>Step 6: Access the Application</h3>
    <p>Open your web browser and go to the URL displayed in the terminal, typically:</p>
    <pre><code>http://127.0.0.1:5000</code></pre>
    <h2>Additional Information</h2>
    <p>For more details, refer to the project documentation or the <code>README.md</code> file.</p>

<p align="center"> <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=24&duration=4000&pause=500&color=36BCF7&center=true&vCenter=true&width=435&lines=Written+with+❤️+by+Aindrik" alt="Typing SVG"> </p>
