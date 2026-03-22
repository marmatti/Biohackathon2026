# FilamentSimulator Config Reference

This config controls one synthetic fluorescence movie: image size, cell scene, filament event and motion, optics, noise, and output files.

## Top-level

### `seed`
Random seed for reproducible sampling of cells, filament events, motion, curvature, and noise.

---

## `movie`

### `n_frames`
Number of frames in the movie.

### `height`
Image height in pixels.

### `width`
Image width in pixels.

---

## `output`

### `movie_npy_name`
Filename for the noisy movie stack saved as a NumPy array.

### `clean_movie_npy_name`
Filename for the clean movie stack before background and noise.

### `movie_labels_json_name`
Filename for movie-level summary labels.

### `frame_labels_json_name`
Filename for frame-level labels.

### `noisy_tif_dir`
Directory for ordered noisy TIF frames.

### `clean_tif_dir`
Directory for ordered clean TIF frames.

---

## `cells`

### `n_cells`
Fixed number of cells placed in each movie.

### `mother_radius_px_range`
Range of mother-cell radii in pixels; sampled independently per cell.

### `bud_probability`
Probability that a sampled cell has a bud.

### `bud_radius_ratio_range`
Range of bud-to-mother radius ratios for budding cells.

### `ellipticity_ratio_range`
Range of ellipse axis ratios used to add slight ellipticity to mother and bud shapes.

### `intensity_range`
Range of diffuse cell fluorescence intensities sampled independently per cell.

### `edge_softness_px_range`
Range controlling how diffuse or sharp the cell boundaries appear.

### `placement_margin_px_range`
Range of margins used to keep sampled cell centers away from the image border.

### `min_center_distance_scale_range`
Range controlling spacing between cell centers relative to summed radii; values below `1.0` allow overlap, around `1.0` give touching cells, above `1.0` enforce separation.

---

## `filament_event`

### `probability_any_filament`
Probability that a movie contains a filament event.

### `total_duration_fraction_range`
Range of fractions of `n_frames` used to sample the total filament event duration.

---

## `filament`

### `length_px_range`
Range of filament lengths in pixels.

### `radius_px_range`
Range of filament tube radii in pixels; sampled once per filament.

### `intensity_mean`
Mean total filament intensity before blur and noise.

### `intensity_std`
Standard deviation of sampled filament intensity.

### `z_center_range_px`
Allowed range of filament center positions along the axial (`z`) direction.

### `rotation_sigma_deg_per_frame`
Standard deviation of random filament rotation per frame, in degrees.

### `translation_sigma_xyz_px`
Standard deviation of random filament translation per frame in `[x, y, z]` pixels.

### `n_centerline_points`
Number of sampled points used to represent the filament centerline.

### `bend_count_range`
Range for the number of bend components sampled per filament.

### `curvature_amplitude_px_range`
Range of bend amplitudes in pixels for each sampled bend component.

### `curvature_drift_sigma_px`
Standard deviation of frame-to-frame drift in bend amplitude.

---

## `optics`

### `psf_sigma_xy_px`
Gaussian blur width in the image plane, approximating the microscope PSF.

### `defocus_sigma_z_px`
Controls how strongly filament intensity decays away from the focal plane in `z`.

---

## `noise`

### `background_offset`
Constant baseline intensity added to the whole image.

### `background_gradient_max`
Maximum magnitude of the random low-frequency background gradient.

### `poisson_scale`
Scale factor used for Poisson shot noise.

### `gaussian_read_sigma`
Standard deviation of additive Gaussian read noise.

### `clip_min`
Minimum allowed pixel value after noise is added.

### `clip_max`
Maximum allowed pixel value after noise is added.
