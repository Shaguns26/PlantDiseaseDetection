from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import uvicorn

# ==========================================
# 1. DEFINE THE MODEL ARCHITECTURE
# (Must match your training code EXACTLY)
# ==========================================
class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.flatten = nn.Flatten()
        # Note: If you changed image size or architecture, update this math
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ==========================================
# 2. CONFIGURATION & REMEDIES
# ==========================================

# IMPORTANT: This list must match the 'new_classes' list from your notebook exactly!
CLASS_NAMES = [
    "Apple___Apple_scab", 
    "Apple___Black_rot", 
    "Apple___Cedar_apple_rust", 
    "Apple___healthy", 
    "Potato___Early_blight", 
    "Potato___Late_blight", 
    "Potato___healthy"
]

# The "Brain" + "Advice" Combo
REMEDIES = {
    "Apple___Apple_scab": {
        "description": "Fungal disease causing dark, scabby spots on fruit and leaves.",
        "remedy": "Rake up and destroy fallen leaves. Apply fungicides like captan or sulfur."
    },
    "Apple___Black_rot": {
        "description": "Causes firm, rotting spots on fruit and reddish-brown spots on leaves.",
        "remedy": "Prune out dead wood and cankers. Remove mummified fruit from the tree."
    },
    "Apple___Cedar_apple_rust": {
        "description": "Bright orange spots on leaves.",
        "remedy": "Remove nearby juniper/cedar plants (the host). Apply fungicide in spring."
    },
    "Potato___Early_blight": {
        "description": "Target-shaped bullseye spots on lower leaves.",
        "remedy": "Improve air circulation. Water at the base, not leaves. Use copper-based fungicide."
    },
    "Potato___Late_blight": {
        "description": "Large, dark brown blotches with white fungal growth.",
        "remedy": "Highly contagious! Remove and destroy infected plants immediately. Do not compost."
    },
    "Potato___healthy": {
        "description": "Your plant looks healthy!",
        "remedy": "Keep up the consistent watering and sunlight."
    },
    "Apple___healthy": {
        "description": "Your plant looks healthy!",
        "remedy": "Ensure it gets good sunlight and prune annually."
    }
}

# ==========================================
# 3. LOAD MODEL & SETUP APP
# ==========================================
app = FastAPI()

# Allow your Lovable app to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your Lovable URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model logic
device = torch.device("cpu") # Hosting is usually CPU unless you pay extra
model = PlantDiseaseCNN(num_classes=len(CLASS_NAMES))

try:
    # Load weights. map_location='cpu' is vital if you trained on GPU!
    model.load_state_dict(torch.load("plant_disease_model.pth", map_location=device))
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Preprocessing (MUST Match Training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ==========================================
# 4. THE API ENDPOINT
# ==========================================
@app.get("/")
def home():
    return {"message": "Plant Doctor API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read Image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # 2. Transform Image
    tensor = transform(image).unsqueeze(0) # Add batch dimension (1, 3, 128, 128)
    
    # 3. Predict
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
        
    # 4. Get Result
    class_name = CLASS_NAMES[class_idx]
    result = REMEDIES.get(class_name, {
        "description": "Unknown disease",
        "remedy": "Consult a local botanist."
    })
    
    return {
        "diagnosis": class_name,
        "description": result["description"],
        "remedy": result["remedy"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
