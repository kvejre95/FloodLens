'''
Code to process FloodNet dataset folders
'''

import os
import zipfile

def extract_and_merge(zip_files, output_dir):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # List of all files and directories in the zip file
            zip_names = zip_ref.namelist()

            for name in zip_names:
                # Extract the individual file or directory
                zip_ref.extract(name)

                # Construct the destination path
                dest_path = os.path.join(output_dir, name.split('/', 1)[-1])

                # If it's a directory, just ensure it exists in the output directory
                if name.endswith('/'):
                    if not os.path.exists(dest_path):
                        os.makedirs(dest_path)
                else:
                    # If it's a file, move it to the corresponding directory in the output directory
                    with zip_ref.open(name) as source, open(dest_path, 'wb') as target:
                        # Copy contents
                        target.write(source.read())

if __name__ == '__main__':
    # List of zip files
    zip_files = ['FloodNet-Supervised_v1.0-20231014T142130Z-001.zip', 'FloodNet-Supervised_v1.0-20231014T142130Z-002.zip', 
                 'FloodNet-Supervised_v1.0-20231014T142130Z-003.zip', 'FloodNet-Supervised_v1.0-20231014T142130Z-004.zip', 
                 'FloodNet-Supervised_v1.0-20231014T142130Z-005.zip', 'FloodNet-Supervised_v1.0-20231014T142130Z-006.zip', 
                 'FloodNet-Supervised_v1.0-20231014T142130Z-007.zip']

    # Output directory where merged content will be saved
    output_dir = 'merged_directory'

    # Extract and merge
    extract_and_merge(zip_files, output_dir)
