import os
import requests
from zipfile import ZipFile

def download_and_extract(url, download_path, extract_path):
    response = requests.get(url)
    os.makedirs(download_path, exist_ok=True)

    with open(os.path.join(download_path, os.path.basename(url)), 'wb') as file:
        file.write(response.content)

    if url.endswith('.zip'):
        with ZipFile(os.path.join(download_path, os.path.basename(url)), 'r') as zip_ref:
            zip_ref.extractall(extract_path)

if __name__ == "__main__":
    # Define download and extract paths
    checkpoints_download_path = 'D:\Internship\Streamlit app\SadTalker\checkpoints'
    enhancer_download_path = 'D:\Internship\Streamlit app\SadTalker\gfpgan\weights'

    # URLs for checkpoint files
    checkpoint_urls = [
        'https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar',
        'https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar',
        'https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors',
        'https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors',
    ]

    # URLs for enhancer files
    enhancer_urls = [
        'https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth',
        'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth',
        'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
        'https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth',
    ]

    # Download and extract checkpoint files
    for url in checkpoint_urls:
        download_and_extract(url, checkpoints_download_path, checkpoints_download_path)

    # Download enhancer files
    for url in enhancer_urls:
        download_and_extract(url, enhancer_download_path, enhancer_download_path)
