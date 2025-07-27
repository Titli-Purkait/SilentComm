# streamlit_app.py

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
import time

from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import av
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1) Constants & labels ---
IMG_SIZE = 48
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# --- 2) Revised Model class with dynamic spatial calculation ---
class EmotionCNN(nn.Module):
    def __init__(self, flattened_dim, hidden_dim, num_classes):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Calculate spatial dimensions based on flattened_dim
        self.spatial_size = flattened_dim // 128
        if self.spatial_size <= 0:
            raise ValueError(f"Invalid spatial size: {self.spatial_size} from flattened_dim {flattened_dim}")
        
        # Find optimal H and W factors
        self.H = int(math.sqrt(self.spatial_size))
        while self.spatial_size % self.H != 0:
            self.H -= 1
        self.W = self.spatial_size // self.H
        
        # Adaptive pooling to ensure consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.H, self.W))
        
        # Fixed flattened dimension
        self.flatten_dim = 128 * self.H * self.W
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_dim, hidden_dim)
        self.bn7 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        
        # Apply adaptive pooling to ensure consistent size
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn7(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)

# --- 3) Robust weight loading for Keras .h5 format ---
@st.cache_resource
def load_model():
    try:
        # Open HDF5 file
        f = h5py.File('model.h5', 'r')
        mw = f['model_weights']
        
        # Find layer groups
        conv_names = sorted([n for n in mw if n.startswith('conv2d_')],
                           key=lambda s: int(s.split('_')[-1]))
        bn_names = sorted([n for n in mw if n.startswith('batchnorm_')],
                         key=lambda s: int(s.split('_')[-1]))
        dense_names = [n for n in mw if n in ['dense_1', 'out_layer']]
        
        # Build model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Determine dimensions from dense_1 layer
        dense_grp = mw['dense_1']
        dense_inner = dense_grp['dense_1'] if 'dense_1' in dense_grp else dense_grp
        kernel = dense_inner['kernel:0'][()]
        flattened_dim = kernel.shape[0]
        hidden_dim = kernel.shape[1]
        
        st.info(f"Loaded dense_1 kernel with shape: {kernel.shape}")
        st.info(f"Using flattened_dim: {flattened_dim}, hidden_dim: {hidden_dim}")
        
        model = EmotionCNN(flattened_dim, hidden_dim, len(CLASS_NAMES)).to(device).eval()
        
        # Helper function to load weights
        def load_layer_weights(layer_group, layer_name, weight_names):
            """Load weights from HDF5 group to PyTorch layer"""
            try:
                grp = layer_group[layer_name]
                inner = grp[layer_name] if layer_name in grp else grp
                return {wn: inner[wn][()] for wn in weight_names if wn in inner}
            except KeyError as e:
                st.error(f"KeyError loading {layer_name}: {str(e)}")
                return {}
            except Exception as e:
                st.error(f"Error loading {layer_name}: {str(e)}")
                return {}

        # Load convolutional layers
        for i, conv_name in enumerate(conv_names):
            conv = getattr(model, f'conv{i+1}')
            weights = load_layer_weights(mw, conv_name, ['kernel:0', 'bias:0'])
            if 'kernel:0' in weights:
                kernel = weights['kernel:0']
                # Keras kernel shape: (H, W, in_c, out_c) -> PyTorch: (out_c, in_c, H, W)
                kernel = np.transpose(kernel, (3, 2, 0, 1))
                conv.weight.data = torch.tensor(kernel, dtype=torch.float32)
            if 'bias:0' in weights:
                conv.bias.data = torch.tensor(weights['bias:0'], dtype=torch.float32)

        # Load batch normalization layers
        for i, bn_name in enumerate(bn_names):
            bn = getattr(model, f'bn{i+1}')
            weights = load_layer_weights(mw, bn_name, 
                                       ['gamma:0', 'beta:0', 'moving_mean:0', 'moving_variance:0'])
            
            if 'gamma:0' in weights:
                bn.weight.data = torch.tensor(weights['gamma:0'], dtype=torch.float32)
            if 'beta:0' in weights:
                bn.bias.data = torch.tensor(weights['beta:0'], dtype=torch.float32)
            if 'moving_mean:0' in weights:
                bn.running_mean = torch.tensor(weights['moving_mean:0'], dtype=torch.float32)
            if 'moving_variance:0' in weights:
                bn.running_var = torch.tensor(weights['moving_variance:0'], dtype=torch.float32)

        # Load dense layers
        # dense_1 layer
        weights = load_layer_weights(mw, 'dense_1', ['kernel:0', 'bias:0'])
        if 'kernel:0' in weights:
            kernel = weights['kernel:0']
            # Keras kernel shape: (in, out) -> PyTorch: (out, in)
            kernel = np.transpose(kernel, (1, 0))
            model.fc1.weight.data = torch.tensor(kernel, dtype=torch.float32)
        if 'bias:0' in weights:
            model.fc1.bias.data = torch.tensor(weights['bias:0'], dtype=torch.float32)
        
        # BatchNorm for dense layer
        if len(bn_names) > 6:  # Check if there's a BN layer for dense
            weights = load_layer_weights(mw, bn_names[6], 
                                       ['gamma:0', 'beta:0', 'moving_mean:0', 'moving_variance:0'])
            if 'gamma:0' in weights:
                model.bn7.weight.data = torch.tensor(weights['gamma:0'], dtype=torch.float32)
            if 'beta:0' in weights:
                model.bn7.bias.data = torch.tensor(weights['beta:0'], dtype=torch.float32)
            if 'moving_mean:0' in weights:
                model.bn7.running_mean = torch.tensor(weights['moving_mean:0'], dtype=torch.float32)
            if 'moving_variance:0' in weights:
                model.bn7.running_var = torch.tensor(weights['moving_variance:0'], dtype=torch.float32)
        
        # Output layer
        weights = load_layer_weights(mw, 'out_layer', ['kernel:0', 'bias:0'])
        if 'kernel:0' in weights:
            kernel = weights['kernel:0']
            kernel = np.transpose(kernel, (1, 0))
            model.fc2.weight.data = torch.tensor(kernel, dtype=torch.float32)
        if 'bias:0' in weights:
            model.fc2.bias.data = torch.tensor(weights['bias:0'], dtype=torch.float32)

        f.close()
        st.success("✅ Model weights loaded successfully!")
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# --- 4) Load model and define transforms ---
MODEL, DEVICE = load_model()
if MODEL is None:
    st.stop()

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# --- 5) Emotion detection from image ---
def detect_emotion(image):
    try:
        image = preprocess(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = MODEL(image)
            probs = F.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
        return CLASS_NAMES[pred.item()], conf.item()
    except Exception as e:
        logging.error(f"Error in detect_emotion: {str(e)}")
        return "Error", 0.0

# --- 6) Enhanced Video Processor with visible annotations ---
class EmotionVideoProcessor(VideoProcessorBase):
    def __init__(self):
        super().__init__()
        self.last_prediction = ("", 0.0)
        self.last_update = 0
        try:
            # Try to load a larger font
            self.font = ImageFont.truetype("arial.ttf", 24)
        except:
            try:
                # Fallback to default font
                self.font = ImageFont.load_default()
                self.font.size = 24
            except:
                # Final fallback
                self.font = ImageFont.load_default()
    
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            # Convert to PIL image
            img = frame.to_image()
            
            # Only process every 10 frames to reduce latency
            current_time = time.time()
            if current_time - self.last_update > 0.1:  # 10 FPS
                # Convert to grayscale for processing
                img_gs = img.convert('L')
                
                # Detect emotion
                emotion, conf = detect_emotion(img_gs)
                self.last_prediction = (emotion, conf)
                self.last_update = current_time
            else:
                emotion, conf = self.last_prediction
            
            # Draw on original color image
            draw = ImageDraw.Draw(img)
            text = f"{emotion} ({conf:.2f})"
            
            # Draw background rectangle
            text_width, text_height = draw.textsize(text, font=self.font)
            draw.rectangle(
                [(10, 10), (20 + text_width, 20 + text_height)],
                fill="black"
            )
            
            # Draw text
            draw.text((15, 15), text, fill="yellow", font=self.font)
            
            # Convert back to video frame
            return av.VideoFrame.from_image(img)
            
        except Exception as e:
            logging.error(f"Error in video processing: {str(e)}")
            return frame

# --- 7) Streamlit UI ---
st.title("Real-time Emotion Detection")
page = st.sidebar.selectbox("Choose Input Type", ["Webcam Live", "Image Upload"])

if page == "Image Upload":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Detect Emotion"):
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            try:
                emotion, conf = detect_emotion(image)
                st.success(f"Predicted Emotion: **{emotion}** (Confidence: {conf:.2%})")
            except Exception as e:
                st.error(f"Error during emotion detection: {str(e)}")

else:  # Webcam Live
    st.subheader("Real-time Webcam Feed")
    st.info("Allow camera access when prompted. Processing may take a moment...")
    
    webrtc_ctx = webrtc_streamer(
        key="emotion-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=EmotionVideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        desired_playing_state=True
    )
    
    if not webrtc_ctx.state.playing:
        st.warning("Waiting for camera to start...")