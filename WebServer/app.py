import streamlit as st
import subprocess
import os
import tempfile
import zipfile
import io
import tifffile
import numpy as np
from PIL import Image

# Paths inside the Docker container
ILASTIK_EXE = "/opt/ilastik/run_ilastik.sh"
PROJECT_FILE = "/app/filament_model_v1.ilp"

st.set_page_config(page_title="Filament Probabilities", layout="wide")

# --- HELPER FUNCTIONS ---
def prepare_for_display(img):
    if img.mode in ("I", "I;16", "F"):
        return img.point(lambda i: i * (1./256)).convert('RGB')
    return img.convert('RGB')

def process_probability_mask(prob_path, class_index, threshold):
    prob_img = tifffile.imread(prob_path)
    
    if prob_img.ndim == 3:
        if prob_img.shape[-1] < prob_img.shape[0]:
            prob_channel = prob_img[:, :, class_index]
        else:
            prob_channel = prob_img[class_index, :, :]
    else:
        prob_channel = prob_img

    total_pixels = prob_channel.size
    confident_pixels = np.count_nonzero(prob_channel >= threshold)
    area_percentage = (confident_pixels / total_pixels) * 100

    prob_channel = np.where(prob_channel >= threshold, prob_channel, 0.0)
    prob_8bit = (prob_channel * 255).astype(np.uint8)
        
    return Image.fromarray(prob_8bit, mode='L'), area_percentage

def create_prob_overlay(original, prob_mask_8bit, opacity):
    base = prepare_for_display(original).convert("RGBA")
    overlay_color = Image.new("RGBA", base.size, (0, 255, 100, 255))
    
    if prob_mask_8bit.size != base.size:
        prob_mask_8bit = prob_mask_8bit.resize(base.size, Image.BILINEAR)
        
    prob_array = np.array(prob_mask_8bit)
    alpha_channel = (prob_array * opacity).astype(np.uint8)
    alpha_mask = Image.fromarray(alpha_channel, mode='L')
    
    overlay_color.putalpha(alpha_mask)
    return Image.alpha_composite(base, overlay_color)

def run_ilastik_with_logs(command):
    """Executes a command and streams the stdout/stderr to the Streamlit UI."""
    # Create an expander for the logs so it doesn't take over the whole screen
    with st.expander("Live Debug Logs (Ilastik Engine)", expanded=True):
        log_box = st.empty()
        logs = []
        
        # Popen allows us to read the output stream while the process is still running
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Merge errors into the standard output stream
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Read line by line as Ilastik prints them
        for line in process.stdout:
            logs.append(line)
            # We keep only the last 100 lines to prevent the browser from lagging
            log_box.code("".join(logs[-100:]), language="bash")
            
        # Wait for the process to officially terminate and grab the exit code
        process.wait()
        
        if process.returncode != 0:
            st.error(f"Ilastik failed with exit code {process.returncode}")
            raise subprocess.CalledProcessError(process.returncode, command, "".join(logs))


# --- SIDEBAR CONTROLS ---
st.sidebar.title("Settings")
processing_mode = st.sidebar.radio("Processing Mode", ["Single Image", "Batch Processing"])
st.sidebar.markdown("---")

target_class = st.sidebar.number_input("Target Class Channel", min_value=0, max_value=5, value=1, step=1)
confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.05)
opacity = st.sidebar.slider("Probability Overlay Opacity", min_value=0.0, max_value=1.0, value=0.6, step=0.05)

# --- MAIN APP UI ---
st.title("Filament Probability App")
st.write("Extract continuous probability maps using Ilastik.")

if processing_mode == "Single Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=False)
    
    if uploaded_file is not None:
        original_img = Image.open(uploaded_file)
        st.subheader("Preview")
        st.image(prepare_for_display(original_img), use_container_width=True)

        if st.button("Run Probability Extraction", type="primary"):
            with st.spinner("Initializing Ilastik..."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    
                    input_filename = f"input_img{os.path.splitext(uploaded_file.name)[1]}"
                    input_path = os.path.join(temp_dir, input_filename)
                    with open(input_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    command = [
                        ILASTIK_EXE, "--headless", f"--project={PROJECT_FILE}",
                        "--export_source=Probabilities", 
                        "--output_format=tiff",
                        f"--output_filename_format={temp_dir}/{{nickname}}_prob.tiff",
                        input_path
                    ]

                    try:
                        # NEW: Call our streaming function instead of subprocess.run
                        run_ilastik_with_logs(command)
                        
                        expected_output_path = os.path.join(temp_dir, "input_img_prob.tiff")
                        
                        if os.path.exists(expected_output_path):
                            prob_mask_8bit, area_pct = process_probability_mask(expected_output_path, target_class, confidence_threshold)
                            blended_img = create_prob_overlay(original_img, prob_mask_8bit, opacity)
                            
                            st.success("Extraction complete!")
                            st.metric(label=f"Filament Area (Confidence ≥ {confidence_threshold})", value=f"{area_pct:.2f}%")
                            st.markdown("---")

                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Probability Heatmap")
                                st.image(prob_mask_8bit, use_container_width=True)
                            with col2:
                                st.subheader("Confidence Overlay")
                                st.image(blended_img, use_container_width=True)
                                
                            with open(expected_output_path, "rb") as file:
                                st.download_button("Download Raw Multi-Channel TIFF", data=file, file_name=f"{os.path.splitext(uploaded_file.name)[0]}_prob.tiff", mime="image/tiff")
                        else:
                            st.error("No output was generated. Check the logs above.")
                    except subprocess.CalledProcessError:
                        st.error("An error occurred during processing.")

else:
    # --- BATCH PROCESSING MODE ---
    uploaded_files = st.file_uploader("Choose images...", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Run Batch Processing", type="primary"):
            with st.spinner(f"Preparing {len(uploaded_files)} images..."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    
                    input_paths = []
                    for i, uploaded_file in enumerate(uploaded_files):
                        ext = os.path.splitext(uploaded_file.name)[1]
                        input_path = os.path.join(temp_dir, f"img_{i}{ext}")
                        with open(input_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        input_paths.append(input_path)

                    command = [
                        ILASTIK_EXE, "--headless", f"--project={PROJECT_FILE}",
                        "--export_source=Probabilities", "--output_format=tiff",
                        f"--output_filename_format={temp_dir}/{{nickname}}_prob.tiff"
                    ]
                    command.extend(input_paths)
                    
                    try:
                        # NEW: Call our streaming function instead of subprocess.run
                        run_ilastik_with_logs(command)
                        
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                            for i, uploaded_file in enumerate(uploaded_files):
                                original_name = os.path.splitext(uploaded_file.name)[0]
                                prob_path = os.path.join(temp_dir, f"img_{i}_prob.tiff")
                                
                                if os.path.exists(prob_path):
                                    zip_file.write(prob_path, f"{original_name}_prob.tiff")
                        
                        st.success("Batch processing complete!")
                        st.download_button("Download All Probability TIFFs (ZIP)", data=zip_buffer.getvalue(), file_name="probability_maps.zip", mime="application/zip", type="primary")
                        
                        # Preview first image
                        first_prob_path = os.path.join(temp_dir, "img_0_prob.tiff")
                        if os.path.exists(first_prob_path):
                            st.subheader("Preview of First Image")
                            orig = Image.open(uploaded_files[0])
                            
                            prob_mask, area_pct = process_probability_mask(first_prob_path, target_class, confidence_threshold)
                            
                            st.metric(label=f"Filament Area (Confidence ≥ {confidence_threshold})", value=f"{area_pct:.2f}%")
                            st.image(create_prob_overlay(orig, prob_mask, opacity), use_container_width=True)

                    except subprocess.CalledProcessError:
                        st.error("An error occurred during batch processing. Check the logs above.")
