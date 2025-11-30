import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import joblib
import os
import tempfile
import time
import torch
from sklearn.ensemble import RandomForestClassifier
from gtts import gTTS
import pygame
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import difflib