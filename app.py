import streamlit as st
import os
from PIL import Image
import torch
from fast_neural_style.transformer_net import TransformerNet
from utils import load_image, save_image, normalize_batch
from torchvision import transforms

# Set directories
CONTENT_DIR = "data/content-images"
STYLE_DIR = "data/style-images"
OUTPUT_DIR = "data/output-images"
MODEL_PATH = "fast_neural_style/download_saved_models.py"

# Ensure directories exist
os.makedirs(CONTENT_DIR, exist_ok=True)
os.makedirs(STYLE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Streamlit app
st.title("🎨 AI-Powered Image Style Transfer")
st.write("Upload a content image and a style image to generate a new stylized image.")



# File upload
content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

if content_file and style_file:
    # Save uploaded images
    content_path = os.path.join(CONTENT_DIR, content_file.name)
    style_path = os.path.join(STYLE_DIR, style_file.name)
    with open(content_path, "wb") as f:
        f.write(content_file.getbuffer())
    with open(style_path, "wb") as f:
        f.write(style_file.getbuffer())

    # Display images
    st.image(content_path, caption="Content Image", width=300)
    st.image(style_path, caption="Style Image", width=300)

    if st.button("Run Style Transfer"):
        with st.spinner("Processing..."):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load content and style images
            content_image = load_image(content_path)
            style_image = load_image(style_path)

            # Define transformations
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255))
            ])

            content_tensor = transform(content_image).unsqueeze(0).to(device)

            # Load model
            style_model = TransformerNet()
            state_dict = torch.load(MODEL_PATH, weights_only= False)
            style_model.load_state_dict(state_dict)
            style_model.to(device)
            style_model.eval()

            # Apply style transfer
            with torch.no_grad():
                output = style_model(content_tensor).cpu()

            # Save output image
            output_image = output[0].clone().clamp(0, 255).numpy()
            output_image = output_image.transpose(1, 2, 0).astype("uint8")
            output_pil = Image.fromarray(output_image)
            output_path = os.path.join(OUTPUT_DIR, f"output_{content_file.name}")
            output_pil.save(output_path)

        st.success("Style Transfer Completed!")
        st.image(output_path, caption="Stylized Image", use_column_width=True)
