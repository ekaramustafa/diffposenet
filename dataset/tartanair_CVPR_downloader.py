import gdown
import zipfile

def download_from_gdrive(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall("monocular_data")
