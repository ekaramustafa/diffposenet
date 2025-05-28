import gdown
import zipfile
import os

def download_from_gdrive(file_id, output_dir, zip_name="monocular_track.zip"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, zip_name)
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        print(f"🔽 Downloading from Google Drive: {url}")
        gdown.download(url, output_path, quiet=False)

        if not os.path.exists(output_path):
            raise FileNotFoundError(f"❌ File was not downloaded: {output_path}")

        print(f"✅ Downloaded: {output_path}")

        # Extract
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            extract_path = os.path.join(output_dir, "monocular_data")
            zip_ref.extractall(extract_path)
            print(f"📂 Extracted to: {extract_path}")

    except Exception as e:
        print(f"❌ Error during download or extraction: {e}")

if __name__ == "__main__":
    download_from_gdrive(
        file_id="1N9BkpQuibIyIBkLxVPUuoB-eDOMFqY8D",
        output_dir="/scratch/users/imelanlioglu21/comp447_project/tartanair_dataset"
    )
