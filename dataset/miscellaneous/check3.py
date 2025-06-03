import os
import torch
from joblib import Parallel, delayed

def check_file(file_path, large_threshold=1e6, small_threshold=-1e6):
    try:
        tensor = torch.load(file_path, map_location='cpu')
        if not torch.is_tensor(tensor):
            return None

        has_nan = torch.isnan(tensor).any().item()
        too_large = (tensor > large_threshold).any().item()
        too_small = (tensor < small_threshold).any().item()

        if has_nan or too_large or too_small:
            return (file_path, has_nan, too_large, too_small)
    except Exception as e:
        return (file_path, 'ERROR', str(e), None)

    return None

def check_normal_flow_files_parallel(root_path, n_jobs=8, large_threshold=1e6, small_threshold=-1e6):
    file_paths = []
    
    for env_name in os.listdir(root_path):
        env_path = os.path.join(root_path, env_name, "Easy")
        if not os.path.isdir(env_path):
            continue

        for p_name in os.listdir(env_path):
            if not p_name.startswith("P"):
                continue

            print(f"🔍 Checking: {env_name}/{p_name}")

            normal_flow_dir = os.path.join(env_path, p_name, "normal_flow")
            if not os.path.isdir(normal_flow_dir):
                print(f"    ⚠️ Skipping: {normal_flow_dir} does not exist")
                continue

            for file in os.listdir(normal_flow_dir):
                if file.endswith(".pt"):
                    file_paths.append(os.path.join(normal_flow_dir, file))

    # Parallel check
    results = Parallel(n_jobs=n_jobs)(
        delayed(check_file)(fp, large_threshold, small_threshold) for fp in file_paths
    )

    # Filter only problematic results
    problem_files = [res for res in results if res is not None]

    # Print results
    if problem_files:
        print("\n🚨 Problematic files found:")
        for path, nan, large, small in problem_files:
            print(f" - {path} | NaN: {nan}, Too Large: {large}, Too Small: {small}")
    else:
        print("✅ All normal_flow .pt files are clean.")

    return problem_files

# Example usage
if __name__ == "__main__":
    check_normal_flow_files_parallel("/kuacc/users/imelanlioglu21/comp447_project/tartanair_dataset/train_data/")
