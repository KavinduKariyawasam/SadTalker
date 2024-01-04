import streamlit as st
import shutil
import torch
from time import strftime
import os, sys
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path

from pydub import AudioSegment
from io import BytesIO
import requests

def download_models():
    st.write("Downloading pre-trained models...")
    os.system("bash scripts/download_models.sh")

def generate_videos(pose_style, uploaded_image, uploaded_audio_files):
    st.session_state.video_index = 0

    result_dir = "D:\\Internship\\Streamlit app\\Streamlit\\Results"
    
    if uploaded_image is not None and uploaded_audio_files is not None and len(uploaded_audio_files) > 0:
        image_path = os.path.join(result_dir, "input_image.jpg")
        with open(image_path, "wb") as image_file:
            image_file.write(uploaded_image.getvalue())
        
        #for index, current_audio_file in enumerate(uploaded_audio_files):
            #audio_path = os.path.join(result_dir, f"input_audio_{index}.mp3")
        for index, audio_path in enumerate(uploaded_audio_files):
            st.write(audio_path)
            
            # Read the content of the audio file into a BytesIO object
            with open(audio_path, "rb") as audio_file:
                audio_content = BytesIO(audio_file.read())

            # Now you have the audio content in a BytesIO object, and you can write it to another location
            new_audio_path = os.path.join(result_dir, f"output_audio_{index}.mp3")
            with open(new_audio_path, "wb") as audio_file:
                audio_file.write(audio_content.getvalue())

            # Define parameters here
            pic_path = image_path
            audio_path = audio_path
            ref_eyeblink = None
            ref_pose = None
            result_dir = './New folder'
            pose_style = pose_style
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            batch_size = 2
            input_yaw_list = None
            input_pitch_list = None
            input_roll_list = None
            checkpoint_dir = './checkpoints'
            size = 256
            expression_scale = 1.
            enhancer = None
            background_enhancer = None
            cpu = False
            face3dvis = False
            still = True
            preprocess = 'full'
            verbose = False
            old_version = False

            # torch.backends.cudnn.enabled = False

            save_dir = os.path.join(result_dir, strftime(f"output_{index}"))
            os.makedirs(save_dir, exist_ok=True)

            current_root_path = os.path.split(sys.argv[0])[0]

            sadtalker_paths = init_path(checkpoint_dir, os.path.join(current_root_path, 'src/config'), size, old_version, preprocess)

            # init model
            preprocess_model = CropAndExtract(sadtalker_paths, device)

            audio_to_coeff = Audio2Coeff(sadtalker_paths,  device)

            animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)

            # crop image and extract 3dmm from image
            first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
            os.makedirs(first_frame_dir, exist_ok=True)

            first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(pic_path, first_frame_dir, preprocess, source_image_flag=True, pic_size=size)
            if first_coeff_path is None:
                st.write("Can't get the coeffs of the input")
                return

            if ref_eyeblink is not None:
                ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
                ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
                os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
                st.write('3DMM Extraction for the reference video providing eye blinking')
                ref_eyeblink_coeff_path, _, _ = preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir, preprocess, source_image_flag=False)
            else:
                ref_eyeblink_coeff_path = None

            if ref_pose is not None:
                if ref_pose == ref_eyeblink: 
                    ref_pose_coeff_path = ref_eyeblink_coeff_path
                else:
                    ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
                    ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
                    os.makedirs(ref_pose_frame_dir, exist_ok=True)
                    st.write('3DMM Extraction for the reference video providing pose')
                    ref_pose_coeff_path, _, _ = preprocess_model.generate(ref_pose, ref_pose_frame_dir, preprocess, source_image_flag=False)
            else:
                ref_pose_coeff_path = None

            # audio2ceoff
            batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=still)
            coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

            # 3dface render
            if face3dvis:
                from src.face3d.visualize import gen_composed_video
                gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))

            # coeff2video
            data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path,
                                        batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                        expression_scale=expression_scale, still_mode=still, preprocess=preprocess, size=size)

            result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info, \
                                        enhancer=enhancer, background_enhancer=background_enhancer, preprocess=preprocess, img_size=size)

            shutil.move(result, save_dir+ ".mp4")
            st.success(f"Video {index + 1} generated successfully!")

            if not verbose:
                shutil.rmtree(save_dir)

            st.session_state.video_index = 0
        st.success("All videos have created successfully.")

    else:
      if uploaded_image is None:
        st.warning("Please upload an portrait image first.")
      else:
        st.write(len(uploaded_audio_files))
        st.warning("Please check the audio files")

# Function to play the next or previous video
def play_video():
    uploaded_audio_files = st.session_state.uploaded_audio_files
    result_dir = "D:\\Internship\\Streamlit app\\Streamlit\\New folder"
    st.markdown(
        f'<style>video {{ width: 100%; max-width: 300px; height: 300px; }}</style>',
        unsafe_allow_html=True
    )

    if st.sidebar.button("Play Previous Video") and st.session_state.video_index > 0:
        st.session_state.video_index -= 1

    if st.sidebar.button("Play Next Video") and st.session_state.video_index < len(uploaded_audio_files) - 1:
        st.session_state.video_index += 1

    if st.session_state.video_index < len(uploaded_audio_files):
        st.markdown(f"Question {st.session_state.video_index + 1}")
        video_path = open(os.path.join(result_dir, f"output_{st.session_state.video_index}.mp4"), 'rb')
        st.video(video_path, format='video/mp4')
    else:
        st.write("All videos are played!")



## Text to voice
def text_to_speech(text, voice_id):
    CHUNK_SIZE = 1024
    url = "https://api.elevenlabs.io/v1/text-to-speech/" + voice_id

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": "f2f4890deef11a7c4f61242f20f101bd"
    }

    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    response = requests.post(url, json=data, headers=headers)
    return response.content


# Main function for the Streamlit app
def main():
    st.set_page_config(
        page_title='Avatar Creation DEMO',
        page_icon='😎',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    st.title("Avatar Creation DEMO")

    voice_dict = {
        "Drew": "29vD33N1CtxCmqQRPOHJ",
        "Clyde": "2EiwWnXFnvU5JabPnv8n",
        "Paul": "5Q0t7uMcjvnagumLfvZi",
        "Dave": "CYw3kZ02Hs0563khs1Fj",
        "Fin": "D38z5RcWu1voky8WS1ja",
        "Antoni": "ErXwobaYiN019PkySvjV",
        "Thomas": "GBv7mTt0atIp3Br8iCZE",
        "Charlie": "IKne3meq5aSn9XLyUdCD",
        "George": "JBFqnCBsd6RMkjVDRZzb",
        "Callum": "N2lVS1w4EtoT3dr4eOWO",
        "Patrick": "ODq5zmih8GrVes37Dizd",
        "Harry": "SOYHLrjzK2X1ezoPC6cr",
        "Liam": "TX3LPaxmHKxFdv7VOQHJ",
        "Josh": "TxGEqnHWrfWFTfGW9XjX",
        "Arnold": "VR6AewLTigWG4xSOukaG",
        "Matthew": "Yko7PKHZNXotIFUBG7I9",
        "James": "ZQe5CZNOzWyzPSCn5a3c",
        "Joseph": "Zlb1dXrM653N07WRdFW3",
        "Jeremy": "bVMeCyTHy58xNoL34h3p",
        "Michael": "flq6f7yk4E4fJM5XTYuZ",
        "Ethan": "g5CIjZEefAph4nQFvHAz",
        "Santa Claus": "knrPHWnBmmDHMoiMeP3l",
        "Daniel": "onwK4e9ZLuTAKqWW03F9",
        "Adam": "pNInz6obpgDQGcFmaJgB",
        "Bill": "pqHfZKP75CvOlQylNhV4",
        "Jessie": "t0jbNlBVZ17f02VDIeMI",
        "Sam": "yoZ06aMxZJJ28mfd3POQ",
        "Giovanni": "zcAOhNBS3c14rBihAFp1",
        "Rachel": "21m00Tcm4TlvDq8ikWAM",
        "Domi": "AZnzlk1XvdvUeBnXmlld",
        "Sarah": "EXAVITQu4vr4xnSDxMaL",
        "Emily": "LcfcDJNUP1GQjkzn1xUU",
        "Elli": "MF3mGyEYCl7XYWbV9V6O",
        "Dorothy": "ThT5KcBeYPX3keUQqHPh",
        "Charlotte": "XB0fDUnXU5powFXDhCwa",
        "Matilda": "XrExE9yKIg1WjnnlVkGX",
        "Gigi": "jBpfuIE2acCO8z3wKNLl",
        "Freya": "jsCqWAovK2LkecY7zXl4",
        "Grace": "oWAxZDx7w5VEj9dCyTzz",
        "Lily": "pFZP5JQG7iQjIQuC4Bku",
        "Serena": "pMsXgVXv3BLzUgSXRplE",
        "Nicole": "piTKgcLEGmPE4e6mEKli",
        "Glinda": "z9fAnlkpzviPz146aGWa",
        "Mimi": "zrHiDhphv9ZnVXBqCLjz"
    }

    # Create or get SessionState
    if 'uploaded_audio_files' not in st.session_state:
        st.session_state.uploaded_audio_files = []
        st.session_state.video_index = 0

    #option = st.sidebar.selectbox('Select an option:', ['Generate Videos', 'Play Videos'])
    rad = st.sidebar.radio("Page Navigation", ["Generate Videos", "Play Videos"])

    if rad == "Generate Videos":
        st.header("Generate Videos")

        st.subheader("Uploade a portrait image HERE:")
        uploaded_image = st.file_uploader("Upload Image HERE", type=["jpg", "jpeg", "png"])

        st.sidebar.header("Audio settings")
        voice_type = st.sidebar.selectbox('Select the gender:', ['Male', 'Female'])

        if voice_type == 'Male':
          voice_selected = st.sidebar.selectbox("Select the voice type:", ['Drew', 'Clyde', 'Paul', 'Dave', 'Fin', 'Antoni', 'Thomas', 'Charlie', 'George', 'Callum', 'Patrick', 'Harry', 'Liam', 'Josh', 'Arnold', 'Matthew', 'James', 'Joseph', 'Jeremy', 'Michael', 'Ethan', 'Santa Claus', 'Daniel', 'Adam', 'Bill', 'Jessie', 'Sam', 'Giovanni'])

        if voice_type == 'Female':
          voice_selected = st.sidebar.selectbox("Select the voice type:", [
              'Rachel', 'Domi', 'Sarah', 'Emily', 'Elli', 'Dorothy', 'Charlotte', 
              'Matilda', 'Gigi', 'Freya', 'Grace', 'Lily', 'Serena', 'Nicole', 'Glinda', 'Mimi'
          ])

        num_voice_files = st.sidebar.number_input("Enter the number of voice files:", min_value=1, value=1, step=1)
        
        pose_style = st.sidebar.number_input("Select the pose style", 0 , 45)
        #still = st.sidebar.checkbox('Still mode')

        if st.sidebar.button("Reset voice clips"):
          st.session_state.uploaded_audio_files = []

        st.subheader("Type the text HERE:")
        # Create text inputs for each voice file
        text_list = []
        for i in range(num_voice_files):
            text_input = st.text_area(f"Enter Text for Voice File {i+1}")
            text_list.append(text_input)
        
        results_directory = "New"
        #st.session_state.uploaded_audio_files = []

        if st.button("Create voice clips"):
            for i, text_input in enumerate(text_list):
                voice_id = voice_dict[voice_selected]
                voice = text_to_speech(text_input, voice_id)
                mp3_file_path = os.path.join(results_directory, f"input_audio_{i}.mp3")
                st.write(mp3_file_path)
                st.session_state.uploaded_audio_files.append(mp3_file_path)
                audio_segment = AudioSegment.from_file(BytesIO(voice), format="mp3")
                audio_segment.export(mp3_file_path, format="mp3")
                st.audio(voice, format="audio/mp3")
            st.write(len(st.session_state.uploaded_audio_files))

        if st.button("Generate Videos"):
            if not (uploaded_image or st.session_state.uploaded_audio_files):
                st.warning("Please upload image and audio files first")
            else:
                st.write(len(st.session_state.uploaded_audio_files))
                generate_videos(pose_style, uploaded_image, st.session_state.uploaded_audio_files)

        #st.video("/content/drive/MyDrive/Streamlit/results/output_0.mp4")
    
    if rad == "Play Videos":
        st.sidebar.header("Play Settings")
        if st.session_state.uploaded_audio_files is not None:
            play_video()
        else:
            st.warning("Please generate videos first.")

if __name__ == '__main__':
    main()