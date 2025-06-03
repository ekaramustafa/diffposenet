import os
from typing import List
import zipfile
import time
from minio import Minio

class TartanAirDownloader(object):
    def __init__(self, dataset_info_path = None, bucket_name = 'tartanair') -> None:
        from minio import Minio
        endpoint_url = "airlab-share-01.andrew.cmu.edu:9000"
        access_key = "4e54CkGDFg2RmPjaQYmW"
        secret_key = "mKdGwketlYUcXQwcPxuzinSxJazoyMpAip47zYdl"
        # these keys were provided from the original repository
        # https://github.com/castacks/tartanair_tools/blob/master/download_training.py
        
        self._isloaded = False
        self._levellist = ['Easy']
        self._typelist = ['image', 'flow']
        self._cameralist = ['left', 'mask']
        self._dataset_info_path = "dataset/datasets.txt"
        if dataset_info_path is not None:
            self._dataset_info_path = dataset_info_path
        
        self.client = Minio(endpoint_url, access_key=access_key, secret_key=secret_key, secure=True)
        self.bucket_name = bucket_name

    
    def load(self, levellist=['Easy'], typelist=['image', 'flow'], cameralist=['left', 'mask']):
        if levellist is None:
            levellist = self._levellist
        if typelist is None:
            typelist = self._typelist
        if cameralist is None:
            cameralist = self._cameralist

        with open(self._dataset_info_path) as f:
            lines = f.readlines()
        zipsizelist = []
        for ll in lines:
            line = ll.strip()      # removes \n, \r, spaces
            if not line:
                continue           # skip empty lines
            parts = line.split()
            if parts and parts[0].endswith('.zip'):
                zipsizelist.append(parts)
        #zipsizelist = [ll.strip().split() for ll in lines if ll.strip().split()[0].endswith('.zip')]
        print("ZIP LIST: ", zipsizelist)

        downloadlist = []
        for zipfile, _ in zipsizelist:
            zf = zipfile.split('/')
            filename = zf[-1]
            difflevel = zf[-2]
        
            filetype = 'flow' if 'flow' in filename else 'image'
            cameratype = next((t for t in ('mask', 'flow') if t in filename), 'left')
            #cameratype = 'flow' if 'flow' in filename else 'left'
        
            if (difflevel in levellist) and (filetype in typelist) and (cameratype in cameralist):
                downloadlist.append(zipfile)
                
        print("Download list: ", downloadlist)
        self.filelist = downloadlist
        self._isloaded = True


    def download(self, destination_path: str, environments: List[str] = ["soulcity"]):
        if not self._isloaded:
            print("The dataset info is not loaded, Please load the dataset info first by calling load()")
            return False, None
    
        # Parse expected sizes from datasets.txt
        expected_sizes = {}
        with open(self._dataset_info_path) as f:
            for line in f:
                if line.strip() and line.strip().split()[0].endswith('.zip'):
                    parts = line.strip().split()
                    expected_sizes[parts[0]] = float(parts[1])  # MB
    
        downloaded_files = []
        
        for source_file_name in self.filelist:
            env_name = source_file_name.split('/')[0]
            if env_name not in environments:
                continue
    
            target_file_name = os.path.join(destination_path, source_file_name.replace('/', '_'))
            expected_size_mb = expected_sizes.get(source_file_name, None)
            print(f"\n📦 Processing {source_file_name} → {target_file_name}")
    
            # Check if the file exists and has the correct size
            if os.path.exists(target_file_name):
                actual_size_mb = os.path.getsize(target_file_name) / (1024 * 1024)
                if expected_size_mb and abs(actual_size_mb - expected_size_mb) < 1.0:
                    print(f"✅ Skipping {target_file_name}, already downloaded and verified.")
                    downloaded_files.append(target_file_name)
                    continue
                else:
                    print(f"⚠️  Partial or corrupt file detected: {target_file_name}. Redownloading.")
                    os.remove(target_file_name)
    
            # Download with retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"⬇️  Attempting download (try {attempt + 1})...")
                    self.client.fget_object(self.bucket_name, source_file_name, target_file_name)
                    actual_size_mb = os.path.getsize(target_file_name) / (1024 * 1024)
                    print(f"📏 Downloaded size: {actual_size_mb:.2f} MB")
    
                    if expected_size_mb and abs(actual_size_mb - expected_size_mb) > 1.0:
                        raise ValueError("Downloaded file size mismatch.")
                    break
                except Exception as e:
                    print(f"❌ Download failed: {e}")
                    if attempt == max_retries - 1:
                        print("❌ Max retries reached. Skipping this file.")
                        continue
                    else:
                        time.sleep(2)  # wait before retry
            else:
                continue  # skip unzip if download failed
    
            downloaded_files.append(target_file_name)
    
        # Unzip downloaded files
        print("\n📂 Unzipping files...")
        extracted_dirs = []
        for zip_path in downloaded_files:
            try:
                if zipfile.is_zipfile(zip_path):
                    extract_dir = zip_path.replace('.zip', '')
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        os.makedirs(extract_dir, exist_ok=True)
                        zip_ref.extractall(extract_dir)
                        print(f"✅ Unzipped {zip_path} → {extract_dir}")
                        extracted_dirs.append(extract_dir)
                else:
                    print(f"❌ Skipped {zip_path}: Not a valid ZIP file.")
            except Exception as e:
                print(f"❌ Error unzipping {zip_path}: {e}")
    
        return True, extracted_dirs


    
    # def download(self, destination_path : str, environments : List[str] = ["abandonedfactory", "soulcity"]):
    #     """
    #     Downloads files from the specified environment and saves them to the given local path.

    #     Args:
    #         destination_path (str): The local directory where downloaded files will be saved.
    #                                 This directory must already exist and be writable.
    #         environments (List[str]): A list representing the environment or source locations 
    #                                 from which files will be downloaded.
    #                                 amusement, oldtown, soulcity, neighborhood, japanesealley, office
    #                                 office2, seasidetown, abandonedfactory, hospital 

    #     Returns:
    #         tuple:
    #             - bool: True if all files were successfully downloaded, False otherwise.
    #             - list[str] or None: A list of file paths for the successfully downloaded files, 
    #                                 or None if an error occurred.

    #     Notes:
    #         - The dataset information must be loaded in advance using the `load()` method.
    #     """

    #     if not self._isloaded:
    #         print("The dataset info is not loaded, Please load the dataset info first by calling load()")
    #         return
        
    #     target_filelist = []

    #     for source_file_name in self.filelist:
    #         env_name = source_file_name.split('/')[0]
    #         if env_name not in environments:
    #             continue   

    #         target_file_name = os.path.join(destination_path, source_file_name.replace('/', '_'))
    #         target_filelist.append(target_file_name)
    #         print('--')
    #         if os.path.isfile(target_file_name):
    #             print('Error: Target file {} already exists..'.format(target_file_name))
    #             return False, None

    #         print(f"  Downloading {source_file_name} from {self.bucket_name}...")
    #         self.client.fget_object(self.bucket_name, source_file_name, target_file_name)
    #         print(f"  Successfully downloaded {source_file_name} to {target_file_name}!")

    #     # Unzip all downloaded files
    #     print("Unzipping downloaded files...")
    #     for zip_path in target_filelist:
    #         try:
    #             if zipfile.is_zipfile(zip_path):
    #                 with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    #                     extract_dir = zip_path.replace('.zip', '')
    #                     os.makedirs(extract_dir, exist_ok=True)
    #                     zip_ref.extractall(extract_dir)
    #                     print(f"  Unzipped {zip_path} to {extract_dir}")
    #             else:
    #                 print(f"  Skipped {zip_path}: Not a valid ZIP file.")
    #         except Exception as e:
    #             print(f"  Error unzipping {zip_path}: {e}")

    #     return True, [zip_path.replace('.zip', '') for zip_path in target_filelist]
    
if __name__ == "__main__":
    downloader = TartanAirDownloader(dataset_info_path="/scratch/users/imelanlioglu21/comp447_project/diffposenet/dataset/datasets.txt")
    downloader.load(levellist=['Easy'], typelist=['image', 'flow'], cameralist=['left','mask'])
    bl, lst = downloader.download("/scratch/users/imelanlioglu21/comp447_project/tartanair_dataset", ["soulcity"])
    print(lst)
