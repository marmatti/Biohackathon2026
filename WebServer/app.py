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
def save_compatible_input(uploaded_file, output_path):
    """
    Reads the uploaded file, ensures it has a 'Channel' dimension for Ilastik, 
    and saves it to the temporary directory.
    """
    try:
        # Reset file pointer just in case
        uploaded_file.seek(0)
        
        # Try reading it as a scientific TIFF
        img_array = tifffile.imread(uploaded_file)
        
        # If the image is exactly 2D (Y, X), Ilastik will crash because it expects (Y, X, C)
        if img_array.ndim == 2:
            # Expand to (Y, X, 1) safely preserving the original bit-depth
            img_array = np.expand_dims(img_array, axis=-1)
            tifffile.imwrite(output_path, img_array)
        else:
            # It already has 3 dimensions (Y, X, C), save the raw bytes to preserve metadata
            uploaded_file.seek(0)
            with open(output_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
    except Exception:
        # Fallback: If tifffile can't read it (e.g., standard PNG/JPG), save raw bytes
        uploaded_file.seek(0)
        with open(output_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

def prepare_for_display(img):
    if img.mode in ("I", "I;16", "F"):
        return img.point(lambda i: i * (1./256)).convert('RGB')
    return img.convert('RGB')

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
    with st.expander("Live Debug Logs (Ilastik Engine)", expanded=True):
        log_box = st.empty()
        logs = []
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
            text=True, bufsize=1, universal_newlines=True
        )
        for line in process.stdout:
            logs.append(line)
            log_box.code("".join(logs[-100:]), language="bash")
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

# --- INITIALIZE SESSION STATE ---
# Single Mode State
if 'single_processed' not in st.session_state:
    st.session_state.single_processed = False
if 'single_data' not in st.session_state:
    st.session_state.single_data = {}

# Batch Mode State
if 'batch_processed' not in st.session_state:
    st.session_state.batch_processed = False
if 'batch_data' not in st.session_state:
    st.session_state.batch_data = {}
if 'batch_zip' not in st.session_state:
    st.session_state.batch_zip = None

# --- MAIN APP UI ---
st.title("Filament Probability App")

if processing_mode == "Single Image":
    st.write("Extract and tune continuous probability maps using Ilastik.")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=False)
    
    if uploaded_file is not None and ('current_single_file' not in st.session_state or st.session_state.current_single_file != uploaded_file.name):
        st.session_state.single_processed = False
        st.session_state.current_single_file = uploaded_file.name

    if uploaded_file is not None:
        original_img = Image.open(uploaded_file)
        
        if not st.session_state.single_processed:
            st.subheader("Preview")
            st.image(prepare_for_display(original_img), use_container_width=True)

            if st.button("Run Probability Extraction", type="primary"):
                with st.spinner("Initializing Ilastik..."):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        input_filename = f"input_img{os.path.splitext(uploaded_file.name)[1]}"
                        input_path = os.path.join(temp_dir, input_filename)
                        # with open(input_path, "wb") as f:
                        #     f.write(uploaded_file.getbuffer())
                        save_compatible_input(uploaded_file, input_path)

                        command = [
                            ILASTIK_EXE, "--headless", f"--project={PROJECT_FILE}",
                            "--export_source=Probabilities", "--output_format=tiff",
                            f"--output_filename_format={temp_dir}/{{nickname}}_prob.tiff",
                            input_path
                        ]

                        try:
                            run_ilastik_with_logs(command)
                            expected_output_path = os.path.join(temp_dir, "input_img_prob.tiff")
                            
                            if os.path.exists(expected_output_path):
                                prob_img = tifffile.imread(expected_output_path)
                                if prob_img.ndim == 3:
                                    prob_channel = prob_img[:, :, target_class] if prob_img.shape[-1] < prob_img.shape[0] else prob_img[target_class, :, :]
                                else:
                                    prob_channel = prob_img

                                with open(expected_output_path, "rb") as file:
                                    tiff_bytes = file.read()

                                st.session_state.single_data = {
                                    "raw_prob": prob_channel,
                                    "orig_img": original_img,
                                    "tiff_bytes": tiff_bytes
                                }
                                st.session_state.single_processed = True
                                st.rerun()
                            else:
                                st.error("No output was generated.")
                        except subprocess.CalledProcessError:
                            st.error("An error occurred during processing.")

        # Single Image Dynamic Rendering
        if st.session_state.single_processed:
            data = st.session_state.single_data
            raw_prob, orig_img = data["raw_prob"], data["orig_img"]
            
            total_pixels = raw_prob.size
            confident_pixels = np.count_nonzero(raw_prob >= confidence_threshold)
            area_pct = (confident_pixels / total_pixels) * 100

            thresholded_prob = np.where(raw_prob >= confidence_threshold, raw_prob, 0.0)
            prob_8bit = (thresholded_prob * 255).astype(np.uint8)
            prob_mask_img = Image.fromarray(prob_8bit, mode='L')
            
            blended_img = create_prob_overlay(orig_img, prob_mask_img, opacity)

            st.success("Extraction complete! Adjust sliders on the left to instantly update.")
            st.metric(label=f"Filament Area (Confidence ≥ {confidence_threshold})", value=f"{area_pct:.2f}%")
            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Probability Heatmap")
                st.image(prob_mask_img, use_container_width=True)
            with col2:
                st.subheader("Confidence Overlay")
                st.image(blended_img, use_container_width=True)
                
            st.download_button("Download Raw TIFF", data=data["tiff_bytes"], file_name=f"{os.path.splitext(uploaded_file.name)[0]}_prob.tiff", mime="image/tiff")
            if st.button("Start Over"):
                st.session_state.single_processed = False
                st.rerun()

else:
    # --- BATCH PROCESSING MODE ---
    st.write("Process multiple images at once and explore the results dynamically.")
    uploaded_files = st.file_uploader("Choose images...", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True)
    
    # Reset batch memory if files change
    if uploaded_files:
        current_file_names = [f.name for f in uploaded_files]
        if 'batch_file_names' not in st.session_state or st.session_state.batch_file_names != current_file_names:
            st.session_state.batch_processed = False
            st.session_state.batch_file_names = current_file_names

    if uploaded_files and not st.session_state.batch_processed:
        if st.button("Run Batch Processing", type="primary"):
            with st.spinner(f"Processing {len(uploaded_files)} images..."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    
                    input_paths = []
                    for i, uploaded_file in enumerate(uploaded_files):
                        ext = os.path.splitext(uploaded_file.name)[1]
                        input_path = os.path.join(temp_dir, f"img_{i}{ext}")
                        # with open(input_path, "wb") as f:
                        #     f.write(uploaded_file.getbuffer())
                        save_compatible_input(uploaded_file, input_path)
                        input_paths.append(input_path)

                    command = [
                        ILASTIK_EXE, "--headless", f"--project={PROJECT_FILE}",
                        "--export_source=Probabilities", "--output_format=tiff",
                        f"--output_filename_format={temp_dir}/{{nickname}}_prob.tiff"
                    ]
                    command.extend(input_paths)
                    
                    try:
                        run_ilastik_with_logs(command)
                        
                        batch_results = {}
                        zip_buffer = io.BytesIO()
                        
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                            for i, uploaded_file in enumerate(uploaded_files):
                                original_name = os.path.splitext(uploaded_file.name)[0]
                                prob_path = os.path.join(temp_dir, f"img_{i}_prob.tiff")
                                
                                if os.path.exists(prob_path):
                                    # Write to ZIP
                                    zip_file.write(prob_path, f"{original_name}_prob.tiff")
                                    
                                    # Load into RAM for exploration
                                    prob_img = tifffile.imread(prob_path)
                                    if prob_img.ndim == 3:
                                        prob_channel = prob_img[:, :, target_class] if prob_img.shape[-1] < prob_img.shape[0] else prob_img[target_class, :, :]
                                    else:
                                        prob_channel = prob_img
                                    
                                    orig_img = Image.open(uploaded_file)
                                    
                                    batch_results[uploaded_file.name] = {
                                        "raw_prob": prob_channel,
                                        "orig_img": orig_img
                                    }
                        
                        # Save everything to session state
                        st.session_state.batch_data = batch_results
                        st.session_state.batch_zip = zip_buffer.getvalue()
                        st.session_state.batch_processed = True
                        st.rerun()

                    except subprocess.CalledProcessError:
                        st.error("An error occurred during batch processing. Check the logs.")

    # Batch Explorer UI
    if st.session_state.batch_processed:
        st.success(f"Successfully processed {len(st.session_state.batch_data)} images!")
        
        # Download ZIP Button
        st.download_button(
            label="Download All Probability TIFFs (ZIP)", 
            data=st.session_state.batch_zip, 
            file_name="batch_probability_maps.zip", 
            mime="application/zip", 
            type="primary"
        )
        st.markdown("---")
        
        # The Interactive Explorer
        st.subheader("Batch Explorer")
        image_names = list(st.session_state.batch_data.keys())
        selected_image = st.selectbox("Select an image to inspect:", image_names)
        
        if selected_image:
            data = st.session_state.batch_data[selected_image]
            raw_prob, orig_img = data["raw_prob"], data["orig_img"]
            
            # Dynamic Math
            total_pixels = raw_prob.size
            confident_pixels = np.count_nonzero(raw_prob >= confidence_threshold)
            area_pct = (confident_pixels / total_pixels) * 100

            thresholded_prob = np.where(raw_prob >= confidence_threshold, raw_prob, 0.0)
            prob_8bit = (thresholded_prob * 255).astype(np.uint8)
            prob_mask_img = Image.fromarray(prob_8bit, mode='L')
            
            blended_img = create_prob_overlay(orig_img, prob_mask_img, opacity)

            # Draw the UI
            st.metric(label=f"Filament Area ({selected_image})", value=f"{area_pct:.2f}%")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(prob_mask_img, use_container_width=True, caption="Probability Heatmap")
            with col2:
                st.image(blended_img, use_container_width=True, caption="Confidence Overlay")
        
        if st.button("Clear Batch & Start Over"):
            st.session_state.batch_processed = False
            st.rerun()