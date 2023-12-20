import os
import streamlit as st

# Function to download pre-trained models
def download_models():
    st.write("Downloading pre-trained models...")

    # Run the script to download models
    os.system("bash scripts/download_models.sh")

# Function to perform inference
def run_inference(source_image_path, driven_audio_path, result_dir, enhancer, preprocess):
    # Construct the inference command
    inference_command = (
        f"python inference.py "
        f"--source_image {source_image_path} "
        f"--driven_audio {driven_audio_path} "
        f"--result_dir {result_dir} "
        f"--enhancer {enhancer} "
        f"--preprocess {preprocess}"
    )

    # Run the inference command
    os.system(inference_command)

# Streamlit app
def main():
    st.title("Machine Learning Model Demo")

    # Add a button to download pre-trained models
    if st.button("Download Models"):
        download_models()
        st.success("Pre-trained models downloaded successfully!")

    # Add Streamlit widgets for user input
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    uploaded_audio = st.file_uploader("Upload Audio", type=["mp3"])

    result_dir = "/content/"  # You may need to customize this based on your model

    # Add a button to trigger the model inference
    if st.button("Generate Video"):
        # Check if both image and audio are uploaded
        if uploaded_image is not None and uploaded_audio is not None:
            # Save the uploaded files to local paths
            image_path = os.path.join(result_dir, "input_image.jpg")
            audio_path = os.path.join(result_dir, "input_audio.mp3")

            with open(image_path, "wb") as image_file:
                image_file.write(uploaded_image.getvalue())

            with open(audio_path, "wb") as audio_file:
                audio_file.write(uploaded_audio.getvalue())

            # Perform model inference using the uploaded image and audio
            run_inference(image_path, audio_path, result_dir, enhancer="gfpgan", preprocess="full")

            # Display the generated video
            st.video(os.path.join(result_dir, "output_video.mp4"))
        else:
            st.warning("Please upload both an image and an audio file.")

if __name__ == "__main__":
    main()
