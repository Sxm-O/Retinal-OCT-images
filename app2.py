import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision import transforms
from PIL import Image
import streamlit as st

st.markdown("""
<div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/Sxm-O/Retinal-OCT-images/main/Khon_Kaen_Wittayayon_School_Logo.svg.png" alt="KKW" width="150">
</div>
""", unsafe_allow_html=True)

#set title
st.markdown("""
<h1 style='text-align: center;'>Retinal OCT Image Classification</h1>
<h3 style='text-align: center;'>For Educational Training</h3>
""", unsafe_allow_html=True)
st.text(" ")
image = Image.open('oct 4 types.png')
st.image(image, caption='OCT images for each category')


#set sub header for description
st.subheader("""
There are four categories for classifying OCT images:
- **Choroidal Neovascularization (CNV)** - the growth of abnormal blood vessels in the choroid. The choroid is the blood vessel-filled middle layer of the eye, which lies between the sclera and retina. These abnormal blood vessels can leak fluid and blood, damaging the retina and causing vision loss ([All About Vision](https://www.allaboutvision.com/conditions/choroidal-neovascularization-cnv/)).
- **Diabetic Macular Edema (DME)** - a complication of diabetes caused by fluid accumulation in the macula, which can affect the fovea. The macula is the central portion of the retina, located at the back of the eye, where vision is the sharpest ([Prevent Blindness](https://preventblindness.org/diabetic-macular-edema-dme/)).
- **Multiple Drusen (DRUSEN)** - yellow deposits under the retina, made up of lipids and proteins. Drusen can vary in size—small, medium, and large. Small drusen are common in individuals aged 50 and older without age-related macular degeneration (AMD). However, having many small drusen and larger drusen are often signs of AMD ([American Academy of Ophthalmology](https://www.aao.org/eye-health/diseases/what-are-drusen#:~:text=Drusen%20are%20yellow%20deposits%20under,are%20often%20signs%20of%20AMD)).
- **Normal Retinas (NORMAL)** - retinas that do not show any signs of pathology like CNV, DME, or drusen.
""")

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

