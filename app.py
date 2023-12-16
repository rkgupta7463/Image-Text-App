import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def text_img(image_path, max_new_tokens=50):  # Adjust max_new_tokens as needed
    # Load processor and model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    # Open and process the image
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")

    # Generate text with max_new_tokens
    text = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    # Decode the generated text
    decoded_text = processor.decode(text[0], skip_special_tokens=True)

    return decoded_text
    
## streamlit function for app UI and logics 
def shadow_box_text(text, shadow_color="rgba(0, 0, 0, 0.2)", text_color="black", font_size="20px"):
    # Use HTML and CSS to create a box with text inside
    box_html = f"""
    <div style="
        box-shadow: 0px 0px 10px {shadow_color};
        padding: 20px;
        margin-bottom:10px;
        background-color: white;
        color: {text_color};
        font-size: {font_size};
        border-radius: 10px;
    ">
        {text}
    </div>
    """
    st.markdown(box_html, unsafe_allow_html=True)

def main():
    st.header("🖼️Image To Text Generator💬")

    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        text_gen=text_img(uploaded_file)
        st.header("Genearted text of this image")
        shadow_box_text(text_gen)
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)


if __name__=="__main__":
    main()
