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
from huggingface_hub import snapshot_download

def download_model():
    REPO_ID = 'vinthony/SadTalker-V002rc'
    snapshot_download(repo_id=REPO_ID, local_dir='./checkpoints', local_dir_use_symlinks=True)

def generate_videos(uploaded_image, uploaded_audio_files):
    st.session_state.video_index = 0

    result_dir = "./Results"
    
    if uploaded_image is not None and uploaded_audio_files is not None and len(uploaded_audio_files) > 0:
        image_path = os.path.join(result_dir, "input_image.jpg")
        with open(image_path, "wb") as image_file:
            image_file.write(uploaded_image.getvalue())

        for index, current_audio_file in enumerate(uploaded_audio_files):
            audio_path = os.path.join(result_dir, f"input_audio_{index}.mp3")
            with open(audio_path, "wb") as audio_file:
                audio_file.write(current_audio_file.getvalue())

            # Define parameters here
            pic_path = image_path
            audio_path = audio_path
            ref_eyeblink = None
            ref_pose = None
            result_dir = './results'
            pose_style = 17
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

# Function to play the next or previous video
def play_video():
    uploaded_audio_files = st.session_state.uploaded_audio_files
    result_dir = "New"
    st.markdown(
        f'<style>video {{ width: 100%; max-width: 300px; height: 300px; }}</style>',
        unsafe_allow_html=True
    )

    if st.button("Play Previous") and st.session_state.video_index > 0:
        st.session_state.video_index -= 1

    if st.button("Play Next") and st.session_state.video_index < len(uploaded_audio_files) - 1:
        st.session_state.video_index += 1

    if st.session_state.video_index < len(uploaded_audio_files):
        video_path = os.path.join(result_dir, f"output_{st.session_state.video_index}.mp4")
        st.video(video_path)
    else:
        st.write("All videos are played!")

# Main function for the Streamlit app
def main():
    download_model()
    st.set_page_config(
        page_title='Avatar Creation DEMO',
        page_icon='😎',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    st.title("Avatar Creation DEMO")

    # Create or get SessionState
    if 'uploaded_audio_files' not in st.session_state:
        st.session_state.uploaded_audio_files = None
        st.session_state.video_index = 0

    option = st.sidebar.selectbox('Select an option:', ['Generate Videos', 'Play Videos'])
    
    if option == 'Generate Videos':
        uploaded_image = st.file_uploader("Upload Image HERE", type=["jpg", "jpeg", "png"])
        uploaded_audio_files = st.file_uploader("Upload Audio Files HERE", type=["mp3"], accept_multiple_files=True)
        
        if st.button("Generate Videos"):
            # Store uploaded_audio_files in session state
            st.session_state.uploaded_audio_files = uploaded_audio_files
            generate_videos(uploaded_image, uploaded_audio_files)
    
    elif option == 'Play Videos':
        if st.session_state.uploaded_audio_files is not None:
            play_video()
        else:
            st.warning("Please generate videos first.")

if __name__ == '__main__':
    main()
