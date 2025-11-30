#!/usr/bin/env python3
"""
Simple test script to verify basic functionality of the gesture recognition system
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import joblib
import os

print("[INFO] Starting basic functionality test...")

# Test 1: Check if model file exists and can be loaded
print("\n=== Test 1: Model Loading ===")
try:
    if os.path.exists("gesture_model.pkl"):
        model = joblib.load("gesture_model.pkl")
        print(f"[OK] Model loaded successfully. Type: {type(model)}")
        print(f"[OK] Model classes: {len(model.classes_)} gestures")
        print(f"[OK] Gesture classes: {sorted(model.classes_.tolist())}")
    else:
        print("[ERROR] gesture_model.pkl not found")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")

# Test 2: Check CSV data files
print("\n=== Test 2: Data Files ===")
csv_files = [f for f in os.listdir() if f.endswith("_data.csv")]
print(f"[INFO] Found {len(csv_files)} CSV files")

valid_files = 0
for filename in csv_files[:5]:  # Check first 5 files
    try:
        df = pd.read_csv(filename)
        if "label" in df.columns and df.shape[1] == 64:
            valid_files += 1
            print(f"[OK] {filename}: {df.shape[0]} samples, label: {df['label'].iloc[0]}")
        else:
            print(f"[WARNING] {filename}: Invalid format - {df.shape}")
    except Exception as e:
        print(f"[ERROR] {filename}: {e}")

print(f"[INFO] Valid CSV files: {valid_files}/{len(csv_files)}")

# Test 3: Check camera availability
print("\n=== Test 3: Camera Test ===")
try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"[OK] Camera accessible. Frame shape: {frame.shape}")
        else:
            print("[WARNING] Camera opened but cannot read frames")
        cap.release()
    else:
        print("[ERROR] Cannot open camera")
except Exception as e:
    print(f"[ERROR] Camera test failed: {e}")

# Test 4: MediaPipe initialization
print("\n=== Test 4: MediaPipe Test ===")
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    print("[OK] MediaPipe hands initialized successfully")
except Exception as e:
    print(f"[ERROR] MediaPipe initialization failed: {e}")

# Test 5: Dictionary loading
print("\n=== Test 5: Dictionary Test ===")
try:
    with open("dictionary.txt", "r") as f:
        dictionary = [word.strip().upper() for word in f.readlines()]
    print(f"[OK] Dictionary loaded: {len(dictionary)} words")
    print(f"[INFO] First 10 words: {dictionary[:10]}")
except Exception as e:
    print(f"[ERROR] Dictionary loading failed: {e}")

print("\n=== Test Complete ===")
print("[INFO] If all tests passed, the system should work correctly.")
