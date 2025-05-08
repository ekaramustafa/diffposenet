import os
from typing import List
import zipfile
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
        self._cameralist = ['left', 'flow']
        self._dataset_info_path = "dataset/datasets.txt"
        if dataset_info_path is not None:
            self._dataset_info_path = dataset_info_path
        
        self.client = Minio(endpoint_url, access_key=access_key, secret_key=secret_key, secure=True)
        self.bucket_name = bucket_name

    
    def load(self, levellist=['Easy'], typelist=['image', 'flow'], cameralist=['left', 'flow']):
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
            line = ll.strip()  # removes \n, \r, spaces
            if not line:
                continue  # skip empty lines
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
            cameratype = 'flow' if 'flow' in filename else 'left'
        
            if (difflevel in levellist) and (filetype in typelist) and (cameratype in cameralist):
                downloadlist.append(zipfile)
                
        print("Download list: ", downloadlist)
        self.filelist = downloadlist
        self._isloaded = True

    
    def download(self, destination_path : str, environments : List[str] = ["soulcity", 
                                                                          "japanesealley", "office", "office2", "seasidetown", 
                                                                          "abandonedfactory", "hospital"]):
        """
        Downloads files from the specified environment and saves them to the given local path.

        Args:
            destination_path (str): The local directory where downloaded files will be saved.
                                    This directory must already exist and be writable.
            environments (List[str]): A list representing the environment or source locations 
                                    from which files will be downloaded.
                                    amusement, oldtown, soulcity, neighborhood, japanesealley, office
                                    office2, seasidetown, abandonedfactory, hospital 

        Returns:
            tuple:
                - bool: True if all files were successfully downloaded, False otherwise.
                - list[str] or None: A list of file paths for the successfully downloaded files, 
                                    or None if an error occurred.

        Notes:
            - The dataset information must be loaded in advance using the `load()` method.
        """

        if not self._isloaded:
            print("The dataset info is not loaded, Please load the dataset info first by calling load()")
            return
        
        target_filelist = []

        for source_file_name in self.filelist:
            env_name = source_file_name.split('/')[0]
            if env_name not in environments:
                continue   

            target_file_name = os.path.join(destination_path, source_file_name.replace('/', '_'))
            target_filelist.append(target_file_name)
            print('--')
            if os.path.isfile(target_file_name):
                print('Error: Target file {} already exists..'.format(target_file_name))
                return False, None

            print(f"  Downloading {source_file_name} from {self.bucket_name}...")
            self.client.fget_object(self.bucket_name, source_file_name, target_file_name)
            print(f"  Successfully downloaded {source_file_name} to {target_file_name}!")

        # Unzip all downloaded files
        print("Unzipping downloaded files...")
        for zip_path in target_filelist:
            try:
                if zipfile.is_zipfile(zip_path):
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        extract_dir = zip_path.replace('.zip', '')
                        os.makedirs(extract_dir, exist_ok=True)
                        zip_ref.extractall(extract_dir)
                        print(f"  Unzipped {zip_path} to {extract_dir}")
                else:
                    print(f"  Skipped {zip_path}: Not a valid ZIP file.")
            except Exception as e:
                print(f"  Error unzipping {zip_path}: {e}")

        return True, [zip_path.replace('.zip', '') for zip_path in target_filelist]
    
if __name__ == "__main__":
    downloader = TartanAirDownloader(dataset_info_path="content/diffposenet/dataset/datasets.txt")
    downloader.load(levellist=['Easy'], typelist=['image', 'flow'], cameralist=['left', 'flow'])
    bl, lst = downloader.download("content/tartanair_dataset", ["soulcity",
                                           "japanesealley", "office", "office2", "seasidetown", 
                                            "abandonedfactory", "hospital"])
    print(lst)
