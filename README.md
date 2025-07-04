# 3D CNN 
This is just a sample code from geeks for geeks. I provided the link below
```
https://www.geeksforgeeks.org/video-classification-with-a-3d-convolutional-neural-network/
```

## Prerequisites
Ensure you have Python installed on your system.

## Create a Virtual Environment
To create a virtual environment, run the following command:
```sh
python -m venv venv
```

### Using Multiple Python Versions
If you have multiple Python versions installed, specify Python "your desired version" explicitly if you want to use a specific version
```sh
python -3.10 -m venv venv
```
or
```sh
python3.10 -m venv venv
```

## Activate the Virtual Environment
Activate the virtual environment with the following command:
```sh
.\venv\Scripts\activate
```

## Install Dependencies
Install the required dependencies using:
```sh
pip install -r .\requirements.txt
```

### Pipeline
```
Input Video
    ↓
3D CNN Backbone (with Dropout, Weight Decay, Regularization)
    ↓
Shared Dense Layer (with more Dropout)
    ↓
    ├── Classification Head → suspicious / non-suspicious
    ├── Landmark Regression Head → facial landmarks per frame
    ├── Landmark Motion Head (∆x, ∆y for each landmark)
    └── Landmark Velocity Head (∆x/∆t, ∆y/∆t for each landmark)
    ↓
[Feature Engineering After Prediction]
    → Blink rate (from eyes)
    → Head turns (from nose/ear positions)
    → Gaze shifts (from pupils or eye landmarks)
    ↓
[Analysis / Visualization / Dashboard / Second-stage Modeling]
```

