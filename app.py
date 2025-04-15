import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import requests

# Load pre-trained model
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

# Load class labels
@st.cache_data
def load_labels():
    LABELS_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    response = requests.get(LABELS_URL)
    return response.json()

# Preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# Predict
def predict(image, model, labels):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5 = torch.topk(probabilities, 5)
    results = []
    for idx, score in zip(top5.indices, top5.values):
        label = labels[str(idx.item())][1]
        results.append((label, score.item()))
    return results

# Streamlit App
st.set_page_config(page_title="Image Classifier", page_icon="🧠", layout="centered")
st.title("🧠 Deep Learning Image Classifier")
st.caption("Built with Streamlit + PyTorch + ResNet50")

uploaded_files = st.file_uploader("Upload one or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    model = load_model()
    labels = load_labels()

    for uploaded_file in uploaded_files:
        st.markdown("---")
        st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Classifying..."):
            results = predict(image, model, labels)

        st.subheader("Top 5 Predictions")
        for label, score in results:
            st.write(f"**{label}** — {score*100:.2f}%")
            st.progress(score)

st.markdown("---")
st.markdown("💡 This app uses a ResNet50 model trained on ImageNet.")
st.markdown("Made with ❤️ using Streamlit and PyTorch.")
