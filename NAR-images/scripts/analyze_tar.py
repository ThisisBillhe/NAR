import os
import json
import hashlib
import tarfile

def calculate_md5(file_path):
    """Calculate the MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def count_samples_in_tar(file_path):
    """Count the number of files (samples) in a tar file."""
    with tarfile.open(file_path, "r") as tar:
        return len(tar.getmembers())

def generate_shard_index(directory, output_json):
    """Generate a shard index JSON file for the given directory."""
    shardlist = []
    
    # Iterate over the files in the directory
    for filename in sorted(os.listdir(directory)):
        if filename.startswith("data_") and filename.endswith(".tar"):
            file_path = os.path.join(directory, filename)
            
            # Calculate MD5 checksum
            # md5sum = calculate_md5(file_path)
            
            # Get file size
            filesize = os.path.getsize(file_path)
            
            # Count the number of samples (files) in the tar file
            nsamples = count_samples_in_tar(file_path)
            
            # Add to shardlist
            shardlist.append({
                "url": filename,
                # "md5sum": md5sum,
                "nsamples": nsamples,
                "filesize": filesize
            })
    
    # Create the final JSON structure
    shard_index = {
        "__kind__": "wids-shard-index-v1",
        "wids_version": 1,
        "shardlist": shardlist,
        "name": "t2i_2M"
    }
    
    # Write the JSON structure to the output file
    with open(output_json, "w") as f:
        json.dump(shard_index, f, indent=2)

# Specify the directory containing the tar files and the output JSON file path
directory = "/mnt/petrelfs/heyefei/ZipAR-X/t2i_2M"
output_path = "t2i_2M.json"

# Generate the shard index JSON file
generate_shard_index(directory, output_path)

print(f"Shard index JSON file generated at {output_path}")