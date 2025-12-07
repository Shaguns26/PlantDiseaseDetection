from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os
import requests
import uvicorn

# ==========================================
# 1. DEFINE THE MODEL ARCHITECTURE
# (Must match your training code EXACTLY)
# ==========================================
class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseCNN, self).__init__()

        # Block 1: Conv -> ReLU -> Pool
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Flatten Dimension Calculation:
        # Input 128x128 -> Pool(2) -> 64 -> Pool(2) -> 32 -> Pool(2) -> 16
        # Final shape: 128 channels * 16 * 16
        self.flatten_dim = 128 * 16 * 16
        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.dropout = nn.Dropout(0.5) # Prevents overfitting
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, self.flatten_dim) # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ==========================================
# 2. CONFIGURATION & REMEDIES
# ==========================================

# IMPORTANT: CLASS_NAMES must be alphabetically sorted to match PyTorch ImageFolder behavior
CLASS_NAMES = [
    'Aloe_Anthracnose',
    'Aloe_Healthy',
    'Aloe_LeafSpot',
    'Aloe_Rust',
    'Aloe_Sunburn',
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Cactus_Dactylopius_Opuntia',
    'Cactus_Healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Money_Plant_Bacterial_wilt_disease',
    'Money_Plant_Healthy',
    'Money_Plant_Manganese_Toxicity',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Snake_Plant_Anthracnose',
    'Snake_Plant_Healthy',
    'Snake_Plant_Leaf_Withering',
    'Spider_Plant_Fungal_leaf_spot',
    'Spider_Plant_Healthy',
    'Spider_Plant_Leaf_Tip_Necrosis',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___healthy'
]

REMEDIES = {
    # --- Indoor Plants: Aloe Vera ---
    'Aloe_Anthracnose': {
        "diagnosis": "Aloe Vera Anthracnose",
        "description": "A fungal disease causing small, water-soaked spots that turn brown/black with a purple margin.",
        "remedy": "Stop overhead watering; keep foliage dry. Apply copper-based fungicide or neem oil. Remove badly infected leaves."
    },
    'Aloe_Healthy': {
        "diagnosis": "Aloe Vera Healthy",
        "description": "Your Aloe looks plump and vibrant!",
        "remedy": "Continue providing bright, indirect light and water only when the soil is completely dry."
    },
    'Aloe_LeafSpot': {
        "diagnosis": "Aloe Vera Leaf Spot",
        "description": "Reddish-brown spots on leaves, often caused by fungal or bacterial infection due to excess moisture.",
        "remedy": "Reduce watering immediately. Move to a sunnier spot. Use a baking soda mixture (1 tsp baking soda + 1 gallon water) as a spray."
    },
    'Aloe_Rust': {
        "diagnosis": "Aloe Vera Rust",
        "description": "Yellow or brown pustules on the leaves caused by fungus.",
        "remedy": "Isolate the plant. Remove infected leaves. Treat with a sulfur-based fungicide. Ensure good air circulation."
    },
    'Aloe_Sunburn': {
        "diagnosis": "Aloe Vera Sunburn",
        "description": "Leaves are turning pale, brown, or orange/red due to sudden exposure to harsh direct sunlight.",
        "remedy": "Move the plant to a location with bright but indirect light. It will likely recover its green color within a few weeks."
    },

    # --- Indoor Plants: Cactus ---
    'Cactus_Dactylopius_Opuntia': {
        "diagnosis": "Cactus Cochineal Scale",
        "description": "White, cotton-like masses on the cactus pads. These are scale insects sucking the sap.",
        "remedy": "Isolate the plant. Dab insects with a cotton swab dipped in rubbing alcohol. Wash with insecticidal soap."
    },
    'Cactus_Healthy': {
        "diagnosis": "Cactus Healthy",
        "description": "Your cactus is looking sharp and healthy!",
        "remedy": "Ensure it gets plenty of light and very little water, especially in winter."
    },

    # --- Indoor Plants: Money Plant (Pothos) ---
    'Money_Plant_Bacterial_wilt_disease': {
        "diagnosis": "Money Plant Bacterial Wilt",
        "description": "Stems turning black or dark green and mushy. Leaves wilting despite wet soil.",
        "remedy": "This is difficult to cure. Remove infected parts immediately using sterilized scissors. Repot in fresh, sterile soil."
    },
    'Money_Plant_Healthy': {
        "diagnosis": "Money Plant Healthy",
        "description": "Vibrant green leaves and strong vines.",
        "remedy": "Water when top inch of soil is dry. Mist leaves occasionally for humidity."
    },
    'Money_Plant_Manganese_Toxicity': {
        "diagnosis": "Money Plant Manganese Toxicity",
        "description": "Yellowing edges on older leaves and dark spots, often caused by excess fertilizer or acidic soil.",
        "remedy": "Flush the soil with distilled water to remove salt buildup. Stop fertilizing for a month. Ensure pH is neutral."
    },

    # --- Indoor Plants: Snake Plant ---
    'Snake_Plant_Anthracnose': {
        "diagnosis": "Snake Plant Anthracnose",
        "description": "Brown, sunken lesions on the leaves.",
        "remedy": "Prune infected leaves. Avoid getting water on the leaves when watering. Apply neem oil."
    },
    'Snake_Plant_Healthy': {
        "diagnosis": "Snake Plant Healthy",
        "description": "Tall, firm, and upright leaves.",
        "remedy": "Thrives on neglect. Allow soil to dry completely between waterings. Low light is okay, but bright is better."
    },
    'Snake_Plant_Leaf_Withering': {
        "diagnosis": "Snake Plant Leaf Withering",
        "description": "Leaves are wrinkling or folding, usually a sign of thirst or root rot.",
        "remedy": "Check soil: If bone dry, water deeply. If soggy/smelly, it is root rot—repot immediately into dry succulent mix."
    },

    # --- Indoor Plants: Spider Plant ---
    'Spider_Plant_Fungal_leaf_spot': {
        "diagnosis": "Spider Plant Fungal Leaf Spot",
        "description": "Black or brown spots on leaves, often from misting or high humidity without airflow.",
        "remedy": "Stop misting. Water at the base. Remove worst leaves. Apply fungicide if spreading."
    },
    'Spider_Plant_Healthy': {
        "diagnosis": "Spider Plant Healthy",
        "description": "Leaves are arching and green with clear variegation.",
        "remedy": "Keep soil slightly moist but not soggy. Avoid direct hot sun."
    },
    'Spider_Plant_Leaf_Tip_Necrosis': {
        "diagnosis": "Spider Plant Brown Tips",
        "description": "Tips of leaves turning brown/black. Often caused by fluoride in tap water or dry air.",
        "remedy": "Switch to distilled or rain water. Increase humidity. Trim brown tips with clean scissors."
    },

    # --- Crops: Apple ---
    'Apple___Apple_scab': {
        "diagnosis": "Apple Scab",
        "description": "Olive-green or black velvety spots on leaves and fruit.",
        "remedy": "Rake and destroy fallen leaves. Apply a fungicide containing copper or sulfur early in the season."
    },
    'Apple___Black_rot': {
        "diagnosis": "Apple Black Rot",
        "description": "Firm, rotting spots on fruit (frog-eye leaf spots).",
        "remedy": "Prune out dead wood and cankers. Remove mummified fruit. Treat with captan fungicide."
    },
    'Apple___Cedar_apple_rust': {
        "diagnosis": "Cedar Apple Rust",
        "description": "Bright orange/yellow spots on leaves.",
        "remedy": "Remove nearby juniper/cedar plants if possible. Apply fungicide (Myclobutanil) at bud break."
    },
    'Apple___healthy': {
        "diagnosis": "Apple Healthy",
        "description": "Leaves are green and free of spots.",
        "remedy": "Maintain regular pruning for airflow and monitor for pests."
    },

    # --- Crops: Cherry ---
    'Cherry_(including_sour)___Powdery_mildew': {
        "diagnosis": "Cherry Powdery Mildew",
        "description": "White, powdery fungal growth on leaves and stems.",
        "remedy": "Prune for air circulation. Spray with a mixture of 1 tbsp baking soda + 1 tsp oil + 1 gallon water."
    },
    'Cherry_(including_sour)___healthy': {
        "diagnosis": "Cherry Healthy",
        "description": "Your cherry tree looks great!",
        "remedy": "Fertilize in early spring. Water deeply during dry spells."
    },

    # --- Crops: Pepper ---
    'Pepper,_bell___Bacterial_spot': {
        "diagnosis": "Pepper Bacterial Spot",
        "description": "Small, water-soaked spots on leaves that turn brown.",
        "remedy": "Remove infected plants to prevent spread. Spray healthy plants with copper fungicide. Avoid overhead watering."
    },
    'Pepper,_bell___healthy': {
        "diagnosis": "Pepper Healthy",
        "description": "Vibrant green leaves and strong stems.",
        "remedy": "Peppers love sun! Ensure consistent watering to prevent blossom end rot."
    },

    # --- Crops: Strawberry ---
    'Strawberry___Leaf_scorch': {
        "diagnosis": "Strawberry Leaf Scorch",
        "description": "Purple blotches on leaves that turn brown/crispy.",
        "remedy": "Remove infected leaves. Keep garden weed-free. Apply fungicide if severe."
    },
    'Strawberry___healthy': {
        "diagnosis": "Strawberry Healthy",
        "description": "Healthy, serrated green leaves.",
        "remedy": "Mulch around plants to keep berries off soil. Water regularly."
    },

    # --- Crops: Tomato ---
    'Tomato___Early_blight': {
        "diagnosis": "Tomato Early Blight",
        "description": "Brown spots with concentric rings (bullseye) on lower leaves.",
        "remedy": "Remove lower infected leaves. Stake plants to keep off ground. Apply mulch. Use copper fungicide."
    },
    'Tomato___Late_blight': {
        "diagnosis": "Tomato Late Blight",
        "description": "Large, dark, water-soaked spots on leaves. White mold may appear underneath.",
        "remedy": "Highly contagious! Remove and destroy infected plants immediately (do not compost). Prevent with fungicides."
    },
    'Tomato___healthy': {
        "diagnosis": "Tomato Healthy",
        "description": "Leaves are green and vibrant.",
        "remedy": "Water at the base to keep leaves dry. Support with cages or stakes."
    }
}

# ==========================================
# 3. LOAD MODEL & SETUP APP
# ==========================================
MODEL_URL = "https://github.com/Shaguns26/PlantDiseaseDetection/releases/download/resnet_scaled/model_3_resnet.pth"
MODEL_PATH = "model_3_resnet.pth"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model... this may take a minute.")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

# 1. Ensure model exists locally
download_model()

# 2. Load the model logic
device = torch.device("cpu")
model = PlantDiseaseCNN(num_classes=len(CLASS_NAMES))

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    
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
    
    # Get remedy details
    result = REMEDIES.get(class_name, {
        "diagnosis": "Unknown",
        "description": "Could not identify specific disease.",
        "remedy": "Try consulting a local nursery."
    })
    
    return {
        "class": class_name, # The raw internal class name
        "diagnosis": result["diagnosis"], # The readable name
        "description": result["description"],
        "remedy": result["remedy"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
