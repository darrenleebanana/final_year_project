## Facial Emotion Analysis from Video

This project performs **Face Detection (FD)**, **Face Recognition (FR)**, and **Emotion Detection (ED)** on subjects recorded in a classroom setting. It leverages a pretrained deep learning model to analyze facial emotions frame-by-frame, assigns tracking-based identifiers to individuals, clusters those identities for consistency across frames, and outputs detailed annotations and statistical summaries.

---

## Download and Set Up the Repository

### Option A: Clone with Git

If you have Git installed:

```bash
git clone https://github.com/darrenleebanana/final_year_project.git
cd final_year_project
```

### Option B: Download as ZIP

1. Visit the [project repository](https://github.com/darrenleebanana/final_year_project)
2. Click **Code** → **Download ZIP**
3. Extract it and open the folder:

   ```bash
   cd final_year_project
   ```

---

## Project Structure

```
├── models/              # Pretrained emotion classifier model (DDAMFN trained on AffectNet)
├── networks/            # Contains DDAM.py and MixedFeatureNet.py (backbones of DDAMFN)
├── image_database/      # Stores videos and generated outputs (videos, JSON, Excel)
├── sort.py              # External SORT tracker script
├── track_with_sort.py   # Reads video, detects faces, predicts emotions, assigns individual IDs, annotates video
├── processing.py        # Clusters Individual IDs into Cluster IDs, saves cluster demographics, final outputs
├── jobscript.sh         # Bash script used to run Python files on a GPU cluster
├── environment.yml      # Conda environment definition file
```

---

## System Requirements

* OS: Linux (accessed via PuTTY)
* Python: 3.10
* GPU: Required (CUDA 12.2 is pre-installed as a shared module on the cluster)
* PyTorch with CUDA support (see Setup)

---

## Setup Instructions

Follow these tested steps to set up the environment:

```bash
conda create -n handover python=3.10 -y
conda activate handover

# Install GPU-compatible PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Face detection (InsightFace + ONNX)
pip install insightface onnxruntime-gpu

# Other packages
conda install filterpy -y
conda install pandas openpyxl -y
```

> A pre-written `environment.yml` is included but was not used to test this workflow.

---

## Input Files

Input files are usually classroom recordings in `.mp4` format.
Save them under `image_database/` in subfolders (e.g., different dates or camera angles):

```
image_database/
└── classroom/
    └── 10s.mp4
```

* The `name` variable should match the video filename (without extension).
* The `BASE_DIR` variable should be set to the folder containing the video.

---

## Processing Pipeline

### `track_with_sort.py`

* Reads the original video
* Scans for faces using InsightFace
* Performs emotion prediction using the pretrained DDAMFN model
* Uses the SORT tracker to assign **Individual IDs** to each subject
* Annotates the video with:

  * Individual ID
  * Predicted emotion
  * Emotion confidence levels
* Outputs a video file annotated with Individual IDs

### `processing.py`

* Clusters **Individual IDs** to derive a more stable **Cluster ID** over time
* Saves **demographics** of Cluster IDs, e.g.:

  * % of total frames each Cluster ID is present
  * Which Individual IDs belong to which Cluster
* Annotates video with Cluster IDs
* Outputs:

  * A JSON file detailing bounding boxes, emotions, and confidence levels per cluster per frame
  * An Excel file:

    * Columns: Cluster IDs
    * Rows: Frame numbers
    * Cell: Predicted emotion (blank if confidence is too low)
* Deletes unnecessary files such as the Individual ID video, if specified

---

## Output Files

* `*_individual.mp4`: Annotated with **Individual IDs** and emotions (optional to keep)
* `*_cluster.mp4`: Annotated with **Cluster IDs**
* `*_timestep.json`: Full timeline of detected Cluster IDs per frame:

  * Bounding box
  * Emotion label
  * Confidence scores (7 basic emotions)
* `.xlsx` file: Cluster-by-frame matrix where:

  * Each row is a frame
  * Each column is a Cluster ID
  * Each cell stores the predicted emotion (empty if confidence < threshold)

---

## How to Run

### Step 1: Run `track_with_sort.py`

1. Open the script and set the following in lines 37 & 38:

   ```python
   name = "10s"                             # Name of your video file without '.mp4'
   BASE_DIR = "image_database/classroom"    # Path to folder containing the video
   ```

2. Run the script **either directly or use sbatch**:

   #### Option A: Run locally (in your environment)

   ```bash
   python track_with_sort.py
   ```

   #### Option B: Run on cluster (using `jobscript.sh`)

   Make sure `jobscript.sh` contains the correct script name and environment activation. Then:

   ```bash
   sbatch jobscript.sh
   ```

> After this step, video annotated with **Individual IDs** (`*_individual.mp4`) would be saved in the same input directory. Watch it to identify any irrelevant individuals or false detections, and note down their **Individual ID** values.

---

3. Watch the output video (`*_individual.mp4`) to check for:

   * False positives
   * Individuals to be excluded (e.g., the teacher)

4. Note down their **Individual ID** values to be used in Step 2 below.

---

### Step 2: Run `processing.py`

1. Set the same `name` and `BASE_DIR` as in Step 1, found on lines 9 & 10.

2. Check and modify these variables in the script:

   * Line 8: `fps = 24` (set to video's fps)
   * Line 24:  `CONFIDENCE_THRESHOLD = 60` (emotion confidence threshold to filter weak emotion predictions)
   * Line 378: `rejected_ids = [IDs to exclude]` (Individual IDs that are to be excluded)
   * Line 396: In `files_to_delete`, include or comment out `"individual_output_video"` to delete video annotated with Individual IDs

3. Run the script **either directly or use sbatch** as in Step 1. `processing.py` contains only clustering and video annotations, a GPU would not be needed. 

---

## Extras

* To get the FPS of your video:

  ```bash
  ffprobe -v 0 -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 input.mp4
  ```

* To trim the video for testing:

  ```bash
  ffmpeg -ss 0 -i input.mp4 -t 5 -c copy trimmed.mp4
  ```

* Make sure GPU is used by checking:

  ```python
  import torch
  print(torch.cuda.is_available())
  print(torch.cuda.get_device_name(0))
  ```

---
