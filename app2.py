import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision import transforms
from PIL import Image
import streamlit as st

image = Image.open('Khon_Kaen_Wittayayon_School_Logo.svg.png')

col1, col2, col3 = st.columns([1, 2, 1]) 
with col2:
    st.image(image, width=150)
    
#set title
st.markdown("<h1 style='text-align: center;'>Retinal OCT Image Classification</h1>", unsafe_allow_html=True)
st.text(" ")
image = Image.open('oct 4 types.png')
st.image(image, caption='OCT images for each category')

#set sub header for description
st.subheader("There are four categories for classifying OCT images:\n- Choroidal Neovascularization (CNV)\n- Diabetic Macular Edema (DME)\n- Multiple Drusen (DRUSEN)\n- Normal Retinas (NORMAL)")


#set header
st.markdown("<div style='text-align: left;'><br>Please upload a Retinal OCT image </div>", unsafe_allow_html=True)

# Load model
# Define class names  
class_names = ['Choroidal neovascularization (CNV)', 'Diabetic Macular Edema (DME)', 'Multiple Drusen (DRUSEN)', 'Normal Retinas (NORMAL)']

# Define your model
model = models.mobilenet_v2(pretrained=True)
for param in model.parameters():
    param.requires_grad = True
model.fc = nn.Linear(in_features=512, out_features=4)

# Load the trained model's parameters
model.load_state_dict(torch.load('model/MobileNet(1)10ep.pth', map_location=torch.device('cpu')))
 
# Preprocess the image
transform = transforms.Compose([
    transforms.Resize(256),  
    transforms.CenterCrop(224),           
    transforms.ToTensor()  
])

def classify_image(image):
    # Preprocess the image
    image = transform(image).unsqueeze(0)  # Add unsqueeze to add batch dimension
    
    # Use your model to predict the class
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    # Calculate confidence
    confidence = torch.softmax(outputs, dim=1)[0] * 100
    
    # Get the predicted class label
    predicted_class_index = predicted.item()
    predicted_class_name = class_names[predicted_class_index] if predicted is not None else "Unknown"
    
    # Return the predicted class label and confidence
    return predicted_class_name, confidence[predicted_class_index].item()

# Upload file
uploaded_image = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    
    # Convert the image to RGB if it has an alpha channel (4 channels)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("_Classifying_...")

    # Perform classification
    predicted_class_name, confidence = classify_image(image)
    
    # Display the results
    st.subheader(f"Prediction: _{predicted_class_name}_")
    st.subheader(f"Confidence: _{confidence:.2f}%_")

