import subprocess
import os
import glob

# Paths as they exist INSIDE the Docker container
ILASTIK_EXE = "/opt/ilastik/run_ilastik.sh"
PROJECT_FILE = "/app/filament_model_v1.ilp"
INPUT_DIR = "/app/data/input"
OUTPUT_DIR = "/app/data/output"

def process_images():
    # Find all PNGs in the mounted input directory
    input_images = glob.glob(os.path.join(INPUT_DIR, "*.png"))

    if not input_images:
        print(f"No PNG images found in {INPUT_DIR}. Please check your volume mounts.")
        return

    print(f"Found {len(input_images)} images. Booting Ilastik headless engine...")

    # Build the headless batch-processing command
    # Notice we use run_ilastik.sh on Linux
    command = [
        ILASTIK_EXE,
        "--headless",
        f"--project={PROJECT_FILE}",
        "--export_source=Simple Segmentation",
        "--output_format=png",
        # {nickname} tells Ilastik to keep the original filename and append a suffix
        f"--output_filename_format={OUTPUT_DIR}/{{nickname}}_mask.png" 
    ]
    
    # Append all the image file paths to the end of the command
    command.extend(input_images)

    # Execute the command
    try:
        subprocess.run(command, check=True)
        print(f"Success! Segmented masks saved to {OUTPUT_DIR}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running Ilastik: {e}")

if __name__ == "__main__":
    process_images()
