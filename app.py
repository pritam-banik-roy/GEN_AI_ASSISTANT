import os
import streamlit as st
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
from PIL import Image
import google.generativeai as genai
from octoai.client import OctoAI
from octoai.util import to_file


# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("api_key")
OCTOAI_TOKEN = os.getenv("OCTOAI_TOKEN")

# Initialize OctoAI client
octoai_client = OctoAI(api_key=OCTOAI_TOKEN)

# Configure Google Gemini AI model
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the models
def get_chatbot_model():
    return genai.GenerativeModel('gemini-pro')

def get_image_captioning_model():
    return genai.GenerativeModel('gemini-1.5-flash-latest')

def get_image_caption(model, image):
    response = model.generate_content([image])
    return response.text

def sdxl_text_to_image(prompt):
    image_resp = octoai_client.image_gen.generate_sdxl(
        prompt=prompt
    )
    images = image_resp.images

    if images[0].removed_for_safety:
        return None
    ext = '.jpg'
    while os.path.isfile(prompt + ext):
        split = prompt.split()
        suffix = split[-1]
        i = 1
        if (suffix[0], suffix[-1]) == ('(', ')') and suffix[1:-1].isdigit():
            i += int(suffix[1:-1])
            prompt = ' '.join(split[:-1])
            prompt += f' ({i})'
        else:
            break
    file_name = prompt + ext
    to_file(images[0], file_name)
    return file_name

# Set page title and icon
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom header
st.markdown("<h1 style='text-align: center; color: #007BFF;'>Meet Your AI Guru!</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white;'>Here to Transform Every Task into a Triumph.</p>", unsafe_allow_html=True)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stButton>button {
        color: white;
        background-color: #007BFF;
        border-radius: 10px;
        border: 2px solid #0056b3;
        padding: 0.5em 1.5em;
        font-weight: bold;
        transition: background-color 0.3s ease, color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
        border: 2px solid #0056b3;
    }
    .stTextInput>div>input {
        background-color: #E9ECEF;
        color: #212529;
        border-radius: 10px;
        padding: 0.5em;
        border: 1px solid #0056b3;
    }
    .stTextInput>div>input:focus {
        border-color: #007BFF;
    }
    .stImage {
        border-radius: 15px;
        border: 2px solid #007BFF;
        margin-top: 1em;
    }
    .stSidebar .stButton>button {
        background-color: #0056b3;
        color: white;
    }
    .stSidebar .stButton>button:hover {
        background-color: #004080;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar menu with icons
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #007BFF;'>🤖 AI Assistant</h1>", unsafe_allow_html=True)
    st.markdown("This assistant helps you with AI-powered tasks like chatting, generating image captions, and creating images from text prompts.")
    st.markdown("---")
    selected_option = option_menu(
        "AI Assistant",
        ["ChatBot", "Image Captioning", "Image Generation"],
        icons=["magic", "image", "camera"],
        menu_icon="cast",
        default_index=0
    )

def role_for_streamlit(user_role):
    return 'assistant' if user_role == 'model' else user_role

# Main content based on selected option
if selected_option == 'ChatBot':
    st.title("🖋️ ChatBot")

    model = get_chatbot_model()
    if "chat_history" not in st.session_state:
        st.session_state['chat_history'] = model.start_chat(history=[])

    # Display the chat history
    for message in st.session_state.chat_history.history:
        with st.chat_message(role_for_streamlit(message.role)):    
            st.markdown(message.parts[0].text)

    # Get user input
    user_input = st.chat_input("Type your message:")
    if user_input:
        st.chat_message("user").markdown(user_input)
        response = st.session_state.chat_history.send_message(user_input)
        with st.chat_message("assistant"):
            st.markdown(response.text)

elif selected_option == 'Image Captioning':
    st.title("✨ Image Captioning")

    model = get_image_captioning_model()
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if st.button("Generate Caption"):
        if uploaded_image:
            image = Image.open(uploaded_image)
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)

            caption = get_image_caption(model, image)
            with col2:
                st.info(caption)
        else:
            st.error("Please upload an image.")

elif selected_option == 'Image Generation':
    st.title("🔮 Image Generation")

    user_prompt = st.text_input("Enter the prompt to generate an image:")

    if st.button("Generate Image"):
        if user_prompt:
            generated_image = sdxl_text_to_image(user_prompt)
            if generated_image:
                st.image(generated_image, caption="Generated Image", use_column_width=True)
            else:
                st.error("Image generation failed or was removed for safety.")
        else:
            st.error("Please enter a prompt.")
