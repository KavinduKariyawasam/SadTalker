import shutil
import torch
from time import strftime
import os, sys
import streamlit as st

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path

def download_models():
    st.write("Downloading pre-trained models...")
    os.system("bash scripts/download_models.sh")


def main():
    st.set_page_config(
      page_title='Avatar Creation DEMO',
      page_icon='😎',
      layout='wide',
      initial_sidebar_state='expanded')


    st.title("Avatar Creation DEMO")

    #st.text("Upload an image and several audio files to generate videos corresponding to those audio files")

    # Check if the session state object exists, and create it if not
    if 'index' not in st.session_state:
        st.session_state.index = 0


    # Add Streamlit widgets for user input
    uploaded_image = st.file_uploader("Upload Image HERE", type=["jpg", "jpeg", "png"])
    uploaded_audio_files = st.file_uploader("Upload Audio Files HERE", type=["mp3"], accept_multiple_files=True)
    #uploaded_audio = st.file_uploader("Upload Audio HERE", type=["mp3"])

    result_dir = "/content/drive/MyDrive/Streamlit/Results"  # You may need to customize this based on your model

               
    # Add a button to trigger the model inference
    # if st.button("Generate Video"):
    #     # Check if both image and audio are uploaded
    #     if uploaded_image is not None and uploaded_audio is not None:
    #         # Save the uploaded files to local paths
    #         image_path = os.path.join(result_dir, "input_image.jpg")
    #         audio_path = os.path.join(result_dir, "input_audio.mp3")

    #         with open(image_path, "wb") as image_file:
    #             image_file.write(uploaded_image.getvalue())

    #         with open(audio_path, "wb") as audio_file:
    #             audio_file.write(uploaded_audio.getvalue())
    
    # Add a button to trigger the model inference
    if st.button("Generate Video"):
        # Check if both image and audio files are uploaded
        if uploaded_image is not None and uploaded_audio_files is not None and len(uploaded_audio_files) > 0:
            # Save the uploaded image to a local path
            image_path = os.path.join(result_dir, "input_image.jpg")
            with open(image_path, "wb") as image_file:
                image_file.write(uploaded_image.getvalue())

            # Get the current audio file
            current_audio_file = uploaded_audio_files[st.session_state.index]

            # Save the current audio file to a local path
            audio_path = os.path.join(result_dir, f"input_audio_{st.session_state.index}.mp3")
            with open(audio_path, "wb") as audio_file:
                audio_file.write(current_audio_file.getvalue())

            # Define parameters here
            pic_path = image_path
            audio_path = audio_path
            ref_eyeblink = None
            ref_pose = None
            result_dir = './results'
            pose_style = 0
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            batch_size = 2
            input_yaw_list = None
            input_pitch_list = None
            input_roll_list = None
            checkpoint_dir = './checkpoints'
            size = 256
            expression_scale = 1.
            enhancer = "gfpgan"
            #enhancer = None
            background_enhancer = None
            cpu = False
            face3dvis = False
            still = True
            preprocess = 'extfull'
            verbose = False
            old_version = False

            # torch.backends.cudnn.enabled = False

            save_dir = os.path.join(result_dir, strftime("output"))
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
            #st.write('3DMM Extraction for source image')
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

            shutil.move(result, save_dir+'.mp4')
            #st.write('The generated video is named:', save_dir+'.mp4')
            st.success("Video generated successfully!")

            if not verbose:
                shutil.rmtree(save_dir)
            
            st.markdown(
              f'<style>video {{ width: 100%; max-width: 300px; height: 300px; }}</style>',
              unsafe_allow_html=True
            )
            st.video(os.path.join(result_dir, "output.mp4"))
    # Button to move to the next audio file
    if st.button("Next"):
        # Increment the index
        st.session_state.index += 1
        # Reset index if it goes beyond the number of uploaded audio files
        st.session_state.index %= len(uploaded_audio_files)

if __name__ == '__main__':
    main()
