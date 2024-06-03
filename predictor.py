import numpy as np
import tkinter as tk
from tkinter import filedialog
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

# Define input shape (should match the input shape used in training)
input_shape = (224, 224)
channels = 3  # RGB images

# Path to the model
path_model = 'cv_4.model.best.hdf5'

# List of classes
classes = ["DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS", "MS",
           "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST", "AION", "PT",
           "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP", "OTHER"]

classes_dis = {
    "DR": "Diabetic Retinopathy",
    "ARMD": "Age-Related Macular Degeneration",
    "MH": "Macular Hole",
    "DN": "Dystrophic Neovascularization",
    "MYA": "Myopic Atrophy",
    "BRVO": "Branch Retinal Vein Occlusion",
    "TSLN": "Tilted Disc Syndrome",
    "ERM": "Epiretinal Membrane",
    "LS": "Lattice Synchysis",
    "MS": "Microaneurysms",
    "CSR": "Central Serous Retinopathy",
    "ODC": "Optic Disc Cupping",
    "CRVO": "Central Retinal Vein Occlusion",
    "TV": "Tobacco Vein",
    "AH": "Arterial Hypertension",
    "ODP": "Optic Disc Pallor",
    "ODE": "Optic Disc Edema",
    "ST": "Star",
    "AION": "Anterior Ischemic Optic Neuropathy",
    "PT": "Pigment Tear",
    "RT": "Retinal Tear",
    "RS": "Retinoschisis",
    "CRS": "Coats Retinopathy",
    "EDN": "Endophthalmitis",
    "RPEC": "Retinal Pigment Epithelial Cell Migration",
    "MHL": "Myelinated Nerve Fiber Layer",
    "RP": "Retinitis Pigmentosa",
}

# Load the model
model = load_model(path_model, compile=False)

# Disease explanations
class DiseaseExplanation:
    def __init__(self):
        self.explanations = {
            "DR": "Diabetic Retinopathy can indicate long-term uncontrolled diabetes, which is a risk factor for heart disease and heart failure.",
            "ARMD": "Age-Related Macular Degeneration is associated with aging and cardiovascular risk factors such as hypertension and atherosclerosis, which can lead to heart failure.",
            "DN": "Dystrophic Neovascularization may be a sign of systemic vascular disease, which can affect the heart's blood supply and function.",
            "BRVO": "Branch Retinal Vein Occlusion is often associated with hypertension and vascular diseases, which are risk factors for heart failure.",
            "CSR": "Central Serous Retinopathy has been linked to stress and cortisol levels, which can impact heart health and contribute to heart failure.",
            "CRVO": "Central Retinal Vein Occlusion may indicate systemic vascular disease and hypertension, both of which are linked to heart failure.",
            "AION": "Anterior Ischemic Optic Neuropathy may occur due to reduced blood flow, which can also affect the heart's blood supply and contribute to heart failure.",
            "CRS": "Coats Retinopathy may be associated with hypertension and other cardiovascular risk factors, which can increase the likelihood of heart failure.",
            "RP": "Retinitis Pigmentosa is a genetic condition, but some forms are associated with systemic disorders such as Usher syndrome, which may involve heart abnormalities and increase the risk of heart failure."
        }

    def get_explanation(self, disease):
        return self.explanations.get(disease, "Explanation not available for this disease.")

# Preprocessing function
def preprocess_image(image_path):
    # Load the image
    image = load_img(image_path, target_size=input_shape)
    image = img_to_array(image)

    # Normalize the image (if this was done during training)
    image = image / 255.0

    # Expand dimensions to match the input shape required by the model
    image = np.expand_dims(image, axis=0)

    return image

# Predict function
def predict_image(model, image_path):
    # Preprocess the image
    image = preprocess_image(image_path)

    # Predict the class probabilities
    predictions = model.predict(image)

    return predictions

# Function to get top N predictions
def get_top_n_predictions(predictions, classes, n=3):
    predicted_dict = dict(zip(classes, predictions[0]))
    sorted_predictions = sorted(predicted_dict.items(), key=lambda item: item[1], reverse=True)
    top_n_predictions = sorted_predictions[:n]
    return top_n_predictions

# Function to assess heart failure based on predictions
def assess_heart_failure(top_predictions):
    # Define the conditions for heart failure assessment
    heart_failure_conditions = ["DR", "ARMD", "DN", "BRVO", "CSR", "CRVO", "AION", "CRS", "RP"]

    # Check if any of the top predictions indicate conditions related to heart failure
    for class_name, probability in top_predictions:
        if class_name in heart_failure_conditions:
            # If a condition related to heart failure is detected, return True
            return True, class_name

    # If none of the predictions indicate heart failure, return False
    return False, None

# Create an instance of DiseaseExplanation
disease_explanation = DiseaseExplanation()

# Create a Tkinter root window (it won't be displayed)
root = tk.Tk()
root.withdraw()

# Open file dialog to select an image file
path_image = filedialog.askopenfilename(
    filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
)

if path_image:
    # Predict on the selected image
    predictions = predict_image(model, path_image)

    # Get top 3 predictions
    top_3_predictions = get_top_n_predictions(predictions, classes, n=3)

    # Assess heart failure based on predictions
    heart_failure, condition = assess_heart_failure(top_3_predictions)

    # Print information about top predictions
    print("You are diagnosed with:")
    for i, (class_name, probability) in enumerate(top_3_predictions, 1):
        print(f"{i}. {classes_dis[class_name]}")

    # Print assessment result and reasoning if applicable
    if heart_failure:
        print("Based on the analysis of your eye scan, it appears that you may be at risk of heart failure.")
        print(f"The system has diagnosed you with {classes_dis[condition]}.")
        explanation = disease_explanation.get_explanation(condition)
        print(f"Let me explain how {classes_dis[condition]} can lead to heart failure: {explanation}")
    else:
        print("Based on the analysis of your eye scan, it seems that you are not at immediate risk of heart failure.")
else:
    print("No file selected.")
