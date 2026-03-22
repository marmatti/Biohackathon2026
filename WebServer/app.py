import streamlit as st
import subprocess
import os
import tempfile
import zipfile
import io
import tifffile
import numpy as np
from PIL import Image
import concurrent.futures

# Paths inside the Docker container
ILASTIK_EXE = "/opt/ilastik/run_ilastik.sh"
PROJECT_FILE = "/app/filament_model_v1.ilp"

st.set_page_config(page_title="Filament Probabilities", layout="wide")

# --- HELPER FUNCTIONS ---
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

# --- NEW: TIFF TO PNG PRE-PROCESSING FUNCTIONS ---
def get_global_high(uploaded_files, percentile, max_workers=4):
    """Scans all uploaded TIFF files to find the global high percentile."""
    def process_file(file_bytes):
        try:
            data = tifffile.imread(io.BytesIO(file_bytes))
            return data.flatten()
        except Exception:
            return None

    all_values = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_file, [f.getvalue() for f in uploaded_files])
        for res in results:
            if res is not None:
                all_values.append(res)
    
    if not all_values: return 1.0 # Fallback
    all_values = np.concatenate(all_values)
    return np.percentile(all_values, percentile)

def preprocess_tiff_to_png(file_buffer, global_high, gamma, contrast_factor, bg_fraction):
    """Applies your custom normalization math and returns a 16-bit PNG PIL Image."""
    file_buffer.seek(0)
    data = tifffile.imread(file_buffer)
    
    # Ensure it's 2D for the math (taking first frame if it's a stack)
    if data.ndim > 2:
        data = data[0]

    H, W = data.shape
    h_start = int(H * (1 - bg_fraction))
    w_end = int(W * bg_fraction)
    
    # Background estimation (lower-left region)
    background_region = data[h_start:, :w_end]
    background = np.mean(background_region)

    # Background subtraction + highlight normalization
    norm = (data - background) / (global_high - background + 1e-8)
    norm = np.clip(norm, 0, 1)

    # Gamma and contrast adjustment
    norm = norm ** gamma
    norm = np.clip(norm * contrast_factor, 0, 1)

    # Convert to 16-bit unsigned integer
    norm_16bit = (norm * 65535).astype(np.uint16)
    
    return Image.fromarray(norm_16bit)


# --- SIDEBAR CONTROLS ---
st.sidebar.title("Settings")
processing_mode = st.sidebar.radio("Processing Mode", ["Single Image", "Batch Processing"])
if processing_mode == "Batch Processing":
    max_workers = st.sidebar.number_input("Max Parallel Workers", min_value=1, max_value=32, value=min(4, os.cpu_count() or 4), step=1)
else:
    max_workers = 1
st.sidebar.markdown("---")

st.sidebar.subheader("1. Pre-Processing (Inputs)")
st.sidebar.caption("Adjusting these requires re-running the extraction.")
percentile_high = st.sidebar.number_input("High Percentile", value=99.99, step=0.01)
gamma = st.sidebar.slider("Gamma", 0.5, 3.0, 1.2, 0.1)
contrast_factor = st.sidebar.slider("Contrast Factor", 0.1, 1.5, 0.8, 0.1)
bg_fraction = st.sidebar.number_input("Background Fraction", value=0.0625, format="%.4f") # 1/16

st.sidebar.markdown("---")
st.sidebar.subheader("2. Mask Tuning (Outputs)")
st.sidebar.caption("Updates dynamically without re-running.")
target_class = st.sidebar.number_input("Target Class Channel", min_value=0, max_value=5, value=0, step=1)
confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.05)
opacity = st.sidebar.slider("Probability Overlay Opacity", min_value=0.0, max_value=1.0, value=0.6, step=0.05)


# --- INITIALIZE SESSION STATE ---
for key in ['single_processed', 'single_data', 'batch_processed', 'batch_data', 'batch_zip']:
    if key not in st.session_state:
        st.session_state[key] = False if 'processed' in key else ({} if 'data' in key else None)


# --- MAIN APP UI ---
st.title("Filament Probability App")

if processing_mode == "Single Image":
    st.write("Extract and tune continuous probability maps using Ilastik.")
    uploaded_file = st.file_uploader("Choose a raw .tiff image...", type=["tif", "tiff"], accept_multiple_files=False)
    
    if uploaded_file is not None and ('current_single_file' not in st.session_state or st.session_state.current_single_file != uploaded_file.name):
        st.session_state.single_processed = False
        st.session_state.current_single_file = uploaded_file.name

    if uploaded_file is not None:
        
        # Display the RAW image as a preview
        if not st.session_state.single_processed:
            st.subheader("Raw TIFF Preview")
            raw_preview = tifffile.imread(uploaded_file)
            if raw_preview.ndim > 2: raw_preview = raw_preview[0]
            st.image(prepare_for_display(Image.fromarray(raw_preview)), use_container_width=True)

            if st.button("Pre-process & Run Extraction", type="primary"):
                with st.spinner("Processing..."):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        
                        # 1. Math: Get local high percentile for this single image
                        local_high = get_global_high([uploaded_file], percentile_high)
                        
                        # 2. Pre-process the TIFF into the 16-bit PNG your model expects
                        processed_png = preprocess_tiff_to_png(uploaded_file, local_high, gamma, contrast_factor, bg_fraction)
                        
                        # 3. Save it to the temp folder
                        input_filename = "input_img.png"
                        input_path = os.path.join(temp_dir, input_filename)
                        processed_png.save(input_path)

                        # 4. Run Ilastik on the newly minted PNG
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
                                    "orig_img": processed_png, # Overlay onto the processed PNG, not raw
                                    "tiff_bytes": tiff_bytes
                                }
                                st.session_state.single_processed = True
                                st.rerun()
                            else:
                                st.error("No output was generated.")
                        except subprocess.CalledProcessError:
                            st.error("An error occurred during processing.")

        # Single Image Dynamic Rendering (Dynamic Thresholds)
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

            st.success("Extraction complete! Adjust Output sliders on the left to instantly update.")
            st.metric(label=f"Filament Area (Confidence ≥ {confidence_threshold})", value=f"{area_pct:.2f}%")
            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Probability Heatmap")
                st.image(prob_mask_img, use_container_width=True)
            with col2:
                st.subheader("Confidence Overlay")
                st.image(blended_img, use_container_width=True)
                
            st.download_button("Download Raw TIFF", data=data["tiff_bytes"], file_name="single_prob.tiff", mime="image/tiff")
            if st.button("Start Over"):
                st.session_state.single_processed = False
                st.rerun()

else:
    # --- BATCH PROCESSING MODE ---
    st.write("Process multiple raw images at once and explore the results dynamically.")
    uploaded_files = st.file_uploader("Choose raw .tiff images...", type=["tif", "tiff"], accept_multiple_files=True)
    
    def process_single_batch_item(i, file_name, file_bytes, temp_dir, global_high, gamma, contrast_factor, bg_fraction, target_class):
        try:
            # 1. Preprocess
            processed_png = preprocess_tiff_to_png(io.BytesIO(file_bytes), global_high, gamma, contrast_factor, bg_fraction)
            input_path = os.path.join(temp_dir, f"img_{i}.png")
            processed_png.save(input_path)
            
            # 2. Ilastik
            prob_path = os.path.join(temp_dir, f"img_{i}_prob.tiff")
            command = [
                ILASTIK_EXE, "--headless", f"--project={PROJECT_FILE}",
                "--export_source=Probabilities", "--output_format=tiff",
                f"--output_filename_format={prob_path}",
                input_path
            ]
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if process.returncode != 0:
                return file_name, False, process.stdout
            
            # 3. Postprocess
            if os.path.exists(prob_path):
                prob_img = tifffile.imread(prob_path)
                if prob_img.ndim == 3:
                    prob_channel = prob_img[:, :, target_class] if prob_img.shape[-1] < prob_img.shape[0] else prob_img[target_class, :, :]
                else:
                    prob_channel = prob_img
                
                orig_img = Image.open(input_path)
                orig_img.load()
                with open(prob_path, "rb") as f:
                    prob_bytes = f.read()
                    
                return file_name, True, {
                    "raw_prob": prob_channel,
                    "orig_img": orig_img,
                    "prob_bytes": prob_bytes,
                    "prob_path": prob_path
                }
            return file_name, False, "Output file not found."
        except Exception as e:
            return file_name, False, str(e)
    
    if uploaded_files:
        current_file_names = [f.name for f in uploaded_files]
        if 'batch_file_names' not in st.session_state or st.session_state.batch_file_names != current_file_names:
            st.session_state.batch_processed = False
            st.session_state.batch_file_names = current_file_names

    if uploaded_files and not st.session_state.batch_processed:
        if st.button("Run Batch Processing", type="primary"):
            with st.spinner(f"Processing {len(uploaded_files)} images..."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    
                    # 1. Calculate the TRUE global high across all uploaded files
                    st.toast("Scanning files to find Global High Percentile...")
                    global_high = get_global_high(uploaded_files, percentile_high, max_workers)
                    
                    batch_results = {}
                    zip_buffer = io.BytesIO()
                    errors = []
                    
                    progress_bar = st.progress(0, text="Processing batch images in parallel...")
                    total_files = len(uploaded_files)
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = []
                        for i, f in enumerate(uploaded_files):
                            futures.append(executor.submit(
                                process_single_batch_item, 
                                i, f.name, f.getvalue(), temp_dir, 
                                global_high, gamma, contrast_factor, bg_fraction, target_class
                            ))
                        
                        completed = 0
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                            for future in concurrent.futures.as_completed(futures):
                                file_name, success, result = future.result()
                                completed += 1
                                progress_bar.progress(completed / total_files, text=f"Processed {completed}/{total_files} images")
                                
                                if success:
                                    original_name = os.path.splitext(file_name)[0]
                                    zip_file.writestr(f"{original_name}_prob.tiff", result["prob_bytes"])
                                    batch_results[file_name] = {
                                        "raw_prob": result["raw_prob"],
                                        "orig_img": result["orig_img"]
                                    }
                                else:
                                    errors.append((file_name, result))
                                    
                    if errors:
                        for fname, err in errors:
                            st.error(f"Failed to process {fname}: \\n{err}")

                    st.session_state.batch_data = batch_results
                    st.session_state.batch_zip = zip_buffer.getvalue()
                    st.session_state.batch_processed = True
                    st.rerun()

    # Batch Explorer UI
    if st.session_state.batch_processed:
        st.success(f"Successfully processed {len(st.session_state.batch_data)} images!")
        st.download_button(
            label="Download All Probability TIFFs (ZIP)", 
            data=st.session_state.batch_zip, 
            file_name="batch_probability_maps.zip", 
            mime="application/zip", 
            type="primary"
        )
        st.markdown("---")
        
        st.subheader("Batch Explorer")
        image_names = list(st.session_state.batch_data.keys())
        selected_image = st.selectbox("Select an image to inspect:", image_names)
        
        if selected_image:
            data = st.session_state.batch_data[selected_image]
            raw_prob, orig_img = data["raw_prob"], data["orig_img"]
            
            total_pixels = raw_prob.size
            confident_pixels = np.count_nonzero(raw_prob >= confidence_threshold)
            area_pct = (confident_pixels / total_pixels) * 100

            thresholded_prob = np.where(raw_prob >= confidence_threshold, raw_prob, 0.0)
            prob_8bit = (thresholded_prob * 255).astype(np.uint8)
            prob_mask_img = Image.fromarray(prob_8bit, mode='L')
            
            blended_img = create_prob_overlay(orig_img, prob_mask_img, opacity)

            st.metric(label=f"Filament Area ({selected_image})", value=f"{area_pct:.2f}%")
            col1, col2 = st.columns(2)
            with col1:
                st.image(prob_mask_img, use_container_width=True, caption="Probability Heatmap")
            with col2:
                st.image(blended_img, use_container_width=True, caption="Confidence Overlay")
        
        if st.button("Clear Batch & Start Over"):
            st.session_state.batch_processed = False
            st.rerun()