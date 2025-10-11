# ==============================================================================
#      FINAL, ROBUST PLANT DISEASE ANALYSIS UI (WITH DETAILED OUTPUT)
# ==============================================================================
# This script ensures a detailed 9-parameter analysis for every prediction.
# It uses Gradio for a reliable, advanced UI with sample images and intelligent
# information retrieval, even for less specific disease predictions.
# ==============================================================================

# ------------------------------------------------------------------------------
# STEP 1: SETUP & DEPENDENCIES
# ------------------------------------------------------------------------------
print("STEP 1: Installing Gradio and downloading OpenCV dependencies...")
!pip install -q gradio opencv-python-headless  # opencv-python-headless for Colab
!apt-get -qq install libgl1-mesa-glx

print("\nSTEP 1.2: Cloning dataset from GitHub...")
!git clone -q https://github.com/pratikkayal/PlantDoc-Dataset.git
print("Dataset cloned successfully.")


# ------------------------------------------------------------------------------
# STEP 2: FAST MODEL TRAINING & SETUP
# ------------------------------------------------------------------------------
print("\nSTEP 2: Training a fast demo model...")
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import gradio as gr
from PIL import Image
import cv2 # Import OpenCV for headless environment
import random

# --- Configuration ---
DATASET_PATH = 'PlantDoc-Dataset'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 3

# --- Data Loading ---
print("Loading a subset of data for training...")
full_train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH, validation_split=0.2, subset="training", seed=123,
    image_size=IMAGE_SIZE, batch_size=BATCH_SIZE
)
class_names = full_train_ds.class_names
print(f"Found {len(class_names)} classes.")
train_ds = full_train_ds.take(30)
val_ds = full_train_ds.take(6)

# --- Model Building & Training ---
num_classes = len(class_names)
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    layers.RandomFlip("horizontal"), layers.RandomRotation(0.1),
    layers.Conv2D(32, 3, padding='same', activation='relu'), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'), layers.MaxPooling2D(),
    layers.Flatten(), layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(f"Starting model training for {EPOCHS} epochs...")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
print("Model trained successfully.")


# ------------------------------------------------------------------------------
# STEP 3: CREATE THE ADVANCED ANALYSIS FUNCTION
# ------------------------------------------------------------------------------
print("\nSTEP 3: Preparing the advanced analysis interface...")

# --- Refined Disease Database for better matching ---
disease_database = {
    'apple scab': {
        "pathogen": "Fungus (Venturia inaequalis)",
        "symptoms": "Olive-green to brown spots on leaves, fruit, and twigs. Leaves may curl and fall prematurely. Spots darken with age.",
        "prevention": "Plant resistant varieties. Rake and destroy fallen leaves. Prune for air circulation.",
        "treatment": "Fungicides (myclobutanil, captan) during susceptible periods (bud break to fruit set).",
        "impact": "Reduces fruit yield and quality, weakens tree."
    },
    'apple rust': {
        "pathogen": "Fungus (Gymnosporangium spp.)",
        "symptoms": "Bright orange-yellow spots on leaves that develop tube-like projections on the underside. Can infect fruit.",
        "prevention": "Remove nearby cedar trees (alternate host). Plant resistant apple varieties.",
        "treatment": "Fungicides (myclobutanil) applied when conditions favor rust development.",
        "impact": "Lowers fruit quality and may lead to premature defoliation."
    },
    'corn blight': {
        "pathogen": "Fungus (e.g., Exserohilum turcicum, Bipolaris zeicola)",
        "symptoms": "Long, tan or grayish-green lesions (spots) on leaves, often starting from lower leaves and moving up.",
        "prevention": "Use resistant corn hybrids. Practice crop rotation and tillage to bury residue.",
        "treatment": "Fungicides may be justified in severe cases on susceptible hybrids or for seed production.",
        "impact": "Reduces photosynthetic area, leading to significant yield loss in severe outbreaks."
    },
    'potato early blight': {
        "pathogen": "Fungus (Alternaria solani)",
        "symptoms": "Dark brown spots on older leaves, often with concentric rings (target-like pattern). Leaves may yellow and drop.",
        "prevention": "Use resistant varieties. Practice crop rotation (3-4 years). Proper plant spacing and irrigation.",
        "treatment": "Fungicides (chlorothalonil, mancozeb) applied preventatively, especially during warm, moist periods.",
        "impact": "Can cause premature defoliation, reducing tuber size and yield."
    },
    'potato late blight': {
        "pathogen": "Oomycete (Phytophthora infestans)",
        "symptoms": "Large, irregular, water-soaked lesions on leaves and stems, rapidly turning brown/black. White fuzzy growth on underside in humid conditions.",
        "prevention": "Plant certified disease-free seed. Destroy volunteer potatoes. Ensure good air circulation. Avoid overhead irrigation.",
        "treatment": "Highly destructive. Immediate application of systemic fungicides (e.g., propamocarb, cymoxanil). Remove and destroy infected plant material.",
        "impact": "Extremely high. Can lead to total crop loss within days under favorable conditions."
    },
    'tomato bacterial spot': {
        "pathogen": "Bacteria (Xanthomonas spp.)",
        "symptoms": "Small, dark, water-soaked spots on leaves that become angular with yellow halos. Fruit may have raised, scabby lesions.",
        "prevention": "Use disease-free seed/transplants. Avoid overhead irrigation. Practice crop rotation. Clean tools.",
        "treatment": "Copper-based bactericides can help suppress spread, but control is difficult once established. Prune infected parts.",
        "impact": "Reduces fruit quality and yield. Can cause significant defoliation."
    },
    'healthy': {
        "pathogen": "N/A (No disease detected)",
        "symptoms": "Leaves exhibit uniform green coloration, robust structure, and no visible lesions, spots, or deformities.",
        "prevention": "Continue good agricultural practices: optimal watering, balanced fertilization, adequate sunlight, and regular monitoring for early signs of stress.",
        "treatment": "N/A (Plant is thriving)",
        "impact": "Excellent. Indicates ideal growing conditions and management practices, leading to maximum yield and quality."
    },
    # General entry for unknown diseases or unclassified predictions
    'general disease': {
        "pathogen": "Likely fungal, bacterial, or viral (Further analysis needed)",
        "symptoms": "Visible abnormalities such as spots, discoloration, wilting, or unusual growth patterns.",
        "prevention": "Isolate affected plants. Improve sanitation. Monitor environmental conditions (humidity, temperature).",
        "treatment": "Consider broad-spectrum organic treatments or consult a local agricultural extension for precise diagnosis and recommended action.",
        "impact": "Moderate to high, depending on severity and spread. Early intervention is crucial to minimize losses."
    }
}

def opencv_leaf_detection(image):
    """Enhances leaf detection with contouring and highlights potential disease areas."""
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise for better contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Otsu's thresholding to get a binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Optional: Morphological operations to clean up the thresholded image
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)


    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the largest contour (assumed to be the leaf)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Draw in green for the main leaf body
        cv2.drawContours(img_cv, [largest_contour], -1, (0, 255, 0), 3)

        # Highlight potential disease areas (simple example: darker spots within the contour)
        # This is a very basic simulation. Real detection would be more complex.
        mask = np.zeros(gray.shape, dtype="uint8")
        cv2.drawContours(mask, [largest_contour], -1, 255, -1) # Fill the contour
        disease_area = cv2.bitwise_and(gray, gray, mask=mask)
        _, disease_thresh = cv2.threshold(disease_area, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) # Find dark spots
        disease_contours, _ = cv2.findContours(disease_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for dc in disease_contours:
            if cv2.contourArea(dc) > 50: # Only highlight reasonably sized spots
                # Draw in red for potential disease areas
                cv2.drawContours(img_cv, [dc], -1, (0, 0, 255), 2)


    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

# The core prediction function for Gradio
def analyze_leaf(image):
    img_resized = image.resize(IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_batch)
    predicted_class_raw = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    detected_image = opencv_leaf_detection(image)

    # Determine health status based on prediction
    is_healthy_prediction = any(keyword in predicted_class_raw.lower() for keyword in ['healthy', 'no disease'])
    health_status = "Healthy" if is_healthy_prediction else "Diseased"

    # Try to find a specific disease match in the database
    # Normalize predicted_class_raw for matching
    predicted_class_lower = predicted_class_raw.lower().replace(" leaf", "").replace(" spots", "").strip()

    # Iterate through database keys to find a match
    matched_disease_key = None
    for db_key in disease_database.keys():
        if db_key in predicted_class_lower:
            matched_disease_key = db_key
            break
    
    # If no specific match, default to 'general disease' or 'healthy'
    if matched_disease_key is None:
        if is_healthy_prediction:
            matched_disease_key = 'healthy'
        else:
            matched_disease_key = 'general disease' # Fallback for unlisted diseases

    details = disease_database[matched_disease_key]

    return (
        detected_image,
        health_status,
        predicted_class_raw, # The raw prediction from the model
        f"{confidence*100:.2f}%",
        details["pathogen"],
        details["symptoms"],
        details["prevention"],
        details["treatment"],
        details["impact"]
    )

# ------------------------------------------------------------------------------
# STEP 4: LAUNCH THE GRADIO WEB INTERFACE
# ------------------------------------------------------------------------------
print("\nSTEP 4: Launching the advanced web interface...")
print("="*50)
print("✅ The interactive web page will appear directly below this message.")
print("="*50)

# Create the Gradio interface with a custom theme and layout
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", secondary_hue="lime")) as demo:
    gr.Markdown(
        """
        # 🌿 .Plants AgriDetect
        A service for farmers to automatically detect diseases in plant leaves.
        Upload an image to classify leaves, help take preventive action early, and reduce dependency on manual inspection.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Leaf Image", height=300)
            submit_btn = gr.Button("Analyze Leaf", variant="primary")
            # MODIFIED: Added clickable sample images from the cloned GitHub repo
            gr.Examples(
                examples=[
                    'PlantDoc-Dataset/Apple Scab Leaf/00075aa8-d81a-4184-8541-b692b78d3fa8___FREC_Scab 3335.JPG',
                    'PlantDoc-Dataset/Corn rust leaf/000782e5-246a-49d9-9533-315f4a67897c___FREC_Rust 1754.JPG',
                    'PlantDoc-Dataset/Tomato leaf late blight/0003faa8-4b24-4395-bf38-5f760420f171___RS_Late.B 5123.JPG',
                    'PlantDoc-Dataset/grape leaf/0022d860-d35a-4200-9118-a6a715a13327___RS_GL-9220.JPG'
                ],
                inputs=image_input,
                label="Sample Images for Testing"
            )

        with gr.Column(scale=2):
            image_output = gr.Image(label="OpenCV-based Detection", height=300)
            gr.Markdown("### 📋 Detailed Analysis Report")
            with gr.Row():
                health_status_output = gr.Textbox(label="1. Health Status", interactive=False)
                disease_type_output = gr.Textbox(label="2. Disease Type (AI Prediction)", interactive=False)
            with gr.Row():
                confidence_output = gr.Textbox(label="3. AI Confidence", interactive=False)
                pathogen_output = gr.Textbox(label="4. Pathogen Type", interactive=False)
            symptoms_output = gr.Textbox(label="5. Key Symptoms", lines=3, interactive=False)
            prevention_output = gr.Textbox(label="6. Preventive Measures", lines=3, interactive=False)
            treatment_output = gr.Textbox(label="7. Recommended Treatment", lines=3, interactive=False)
            impact_output = gr.Textbox(label="8. Potential Impact on Crop", lines=2, interactive=False)
            # Added a 9th parameter for cultivation suitability
            cultivation_suitability_output = gr.Textbox(label="9. Cultivation Suitability", lines=2, interactive=False)

    def dynamic_cultivation_suitability(health_status, disease_type):
        if "Healthy" in health_status:
            return "Excellent for cultivation. Continue optimal care practices."
        elif "late blight" in disease_type.lower() or "bacterial spot" in disease_type.lower():
            return "Poor for cultivation. This disease is highly destructive; immediate action is required. Consider isolating or removing affected plants."
        elif "early blight" in disease_type.lower() or "scab" in disease_type.lower() or "rust" in disease_type.lower():
            return "Fair to Moderate suitability. Manageable with timely treatment, but may impact yield. Monitor closely."
        else:
            return "Variable suitability. Depends on severity and specific pathogen. Consult expert for best practices."

    # Update the click function to include the 9th parameter
    submit_btn.click(
        fn=analyze_leaf,
        inputs=image_input,
        outputs=[
            image_output, health_status_output, disease_type_output, confidence_output,
            pathogen_output, symptoms_output, prevention_output, treatment_output, impact_output
        ],
        # Add chain to update cultivation suitability based on other outputs
    ).success(
        fn=dynamic_cultivation_suitability,
        inputs=[health_status_output, disease_type_output],
        outputs=cultivation_suitability_output
    )

# Launch the interface with a public link
demo.launch(share=True, debug=True)