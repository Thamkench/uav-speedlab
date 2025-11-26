# UAV-Speed: Monocular UAV Vehicle Speed Estimation Pipeline

> YOLOv11 + BoT-SORT + Homography-based motion compensation for **vehicle speed estimation** under UAV oblique-view scenarios.

## 🎬 Demo Gallery (UAV speed estimation)

<p align="center">
  <video src="data/output/demo_M0703.mp4" width="32%" controls muted loop></video>
  <video src="data/output/demo_M1003.mp4" width="32%" controls muted loop></video>
  <video src="data/output/demo_M1303.mp4" width="32%" controls muted loop></video>
</p>

This repository provides a **practical, end-to-end pipeline** to estimate vehicle speed from **monocular UAV videos**.  
The pipeline is designed for **daily inspection flights** (low–medium altitude, oblique view), where:

- The UAV is **moving** (camera ego-motion),
- Vehicles may be **stopped** or **moving slowly**,
- We want **robust “static vs moving” classification** and **stable speed display**.

---

## ✨ Key Features

- **Object detection:** YOLOv11 for vehicle/person detection.
- **Multi-object tracking:** BoT-SORT for track IDs across frames.
- **Camera motion compensation:** Homography-based background motion removal.
- **Static vs moving classification:**
  - Geometric evidence: track displacement after homography compensation.
  - Photometric evidence: brightness residual in local patches.
- **Speed estimation (km/h):**
  - Using pixel-to-meter ratio based on a reference object / scene prior.
  - Only displays speed for **non-static vehicles**.
- **Ready-to-use script:**
  - One-line command to run on your own UAV videos.
  - Outputs a speed-annotated demo video.

---

## 🧩 Pipeline Overview

The core pipeline is:

1. **Detection (YOLOv11)**  
   - Input: raw UAV frame  
   - Output: bounding boxes + class scores

2. **Tracking (BoT-SORT)**  
   - Input: current detections + previous tracks  
   - Output: track IDs for each object

3. **Camera Motion Estimation (Homography)**  
   - Use background keypoints between frames to estimate camera motion  
   - Compensate track positions to get **motion relative to the ground**

4. **Static vs Moving Decision**  
   - For each track: accumulate compensated displacements over a time window  
   - Combine geometric + brightness cues with hysteresis to avoid flickering  
   - Output a binary state: `static` / `moving`

5. **Speed Estimation**  
   - Convert pixel displacement → meters using a scene-specific scale  
   - Compute speed in km/h only for `moving` vehicles

6. **Visualization**  
   - Draw bounding boxes + ID + (optional) speed on each frame  
   - Save annotated video to disk

---

## 📁 Repository Structure

A possible structure (adjust to your actual repo):

```text
speed-detection/
  ├─ configs/
  │   ├─ trackers/
  │   │   ├─ botsort.yaml          # BoT-SORT tracker config
  │   │   └─ bytetrack.yaml        # ByteTrack tracker config (optional)
  │   └─ speed_config.yaml         # Speed estimation & status config
  │
  ├─ data/
  │   ├─ videos/                   # Input UAV videos (user-provided)
  │   ├─ output/                   # Speed-annotated demo videos
  │   │   ├─ demo_M0703.mp4
  │   │   ├─ demo_M1003.mp4
  │   │   └─ demo_M1303.mp4
  │   └─ visdrone_frames/          # (Optional) raw VisDrone frames / examples
  │
  ├─ scripts/
  │   ├─ run_speed.py              # Main entry: run full speed pipeline on a video
  │   ├─ run_yolo_detect_video.py  # Only YOLO detection / tracking visualization
  │   └─ frames_to_video.py        # Utility: convert frame folder back to video
  │
  ├─ src/
  │   ├─ config/
  │   │   └─ loader.py             # Config loading & dynamic-classes helpers
  │   ├─ io/
  │   │   └─ frame_source.py       # Video / frame-sequence abstraction
  │   ├─ motion/
  │   │   ├─ homography.py         # Camera motion estimation (ORB + RANSAC)
  │   │   ├─ speed_estimator.py    # Speed estimation helpers (mpp, smoothing)
  │   │   └─ state_machine.py      # Static / moving state machine logic
  │   ├─ pipeline/
  │   │   └─ speed_pipeline.py     # High-level pipeline wrapper
  │   └─ vis/
  │       └─ draw.py               # Drawing bboxes, labels, speed overlay
  │
  ├─ weights/
  │   ├─ yolo11-visdrone.pt        # VisDrone-trained YOLOv11 weights
  │   └─ yolo11.pt                 # (Optional) general YOLOv11 weights
  │
  ├─ README.md                     # English README
  ├─ README_ch.md                  # Chinese README (optional)
  └─ requirements.txt
```

---

## ⚙️ Environment & Installation

This project is intentionally kept simple.  
If you can run Ultralytics YOLO, you can almost certainly run this repo.

### Python & OS

- Python **3.8+** (tested with 3.8–3.11)
- Any OS with Python support:
  - Linux, Windows, or macOS should all work

A GPU is **optional**. If you have an NVIDIA GPU with CUDA, you can get much faster inference;  
otherwise CPU-only mode also works (just slower).

### Recommended setup (optional, but nice)

Using a virtual environment is recommended, but not required. For example with conda:

```bash
conda create -n uav-speed python=3.10 -y
conda activate uav-speed
pip install -r requirements.txt
```

## 🚀 Quick Start

This repository provides a **monocular UAV speed estimation pipeline** based on:

- YOLOv11 object detection
- BoT-SORT multi-object tracking
- Homography-based camera motion compensation

We also provide a YOLO model **fine-tuned on UAV / VisDrone-style data**:

- `weights/yolo11l-visdrone.pt`

You can use this model directly, or swap in your own YOLO checkpoint.  
**Important:** the pipeline removes all *dynamic objects* when estimating camera motion (homography),  
so you **must** tell the config which classes are “moving” and what their approximate lengths are.

---

### 1. Run the pipeline on your own UAV video (with the provided model)

1. Put your UAV video under `data/videos/`, for example:

   ```text
   data/videos/my_uav_video.mp4
   ```

2. Make sure the config points to the provided UAV/VisDrone model and dynamic classes:
- dynamic_classes.names defines which classes are dynamic (cars, trucks, buses, …).
These boxes are removed from the feature-matching mask, so homography is estimated only from background points.

- speed.class_length_m defines the approximate physical length (in meters) of each class.
The pipeline uses this to convert bounding-box size → meters-per-pixel → speed.
3. Run the speed pipeline:
   ```text
   PYTHONPATH=. python scripts/run_speed.py \
    --video data/videos/my_uav_video.mp4 \
    --config configs/speed_config.yaml \
    --out data/output/my_uav_video_speed.mp4
   ```
    The output video will contain:
    - STATIC labels for vehicles detected as static
    - xx.x km/h for vehicles detected as moving (after camera-motion compensation)
    (Optional) speed overlay on the video

### 2. Use your own YOLO model
You can replace the provided UAV model with any YOLOv8 / YOLOv11-style model, as long as:
The model is supported by ultralytics
You correctly configure:
- which classes are dynamic (dynamic_classes)
- the approximate length for each class (speed.class_length_m)

1. Put your YOLO checkpoint under `weights/`, for example:

   ```text
   weights/my_yolo_model.pt
   ```

2. Make sure the config points to your YOLO checkpoint and dynamic classes.
   ````text
   model:
    weights: "weights/my_yolo.pt"

   dynamic_classes:
    names:
    - car
    - truck
    - bus
    # add any other moving classes you want to:
    # - remove from homography estimation
    # - estimate speed for
    ids: []   # optional: use numeric class IDs if you prefer

   speed:
      default_length_m: 5.0

      class_length_m:
      car:          4.5
      truck:        9.0
      bus:          11.0
   ````
3. Run the speed pipeline:
   ```text
   PYTHONPATH=. python scripts/run_speed.py \
    --video data/videos/my_uav_video.mp4 \
    --config configs/speed_config.yaml \
    --out data/output/my_uav_video_speed.mp4
   ```
- If dynamic_classes is **not set correctly** (e.g. you forget to include your “truck” class),
those vehicles will **not be masked** out when estimating homography, and the background motion estimation may degrade.
- **(Optional) Tune static / moving thresholds**
You can also adjust how strict the static vs moving decision is in the same config file:
```text
speed:
  default_length_m: 5.0

  class_length_m:
  car:          4.5
  truck:        9.0
  bus:          11.0

  static_threshold: 0.5
  moving_threshold: 0.5
```

- **(Optional) Speed overlay on the video**
You can also adjust how strict the static vs moving decision is in the same config file:
    ````text
    static_detection:
     d_static_px: 2.5    # max pixel displacement to still be considered static
     d_moving_px: 5.0    # min pixel displacement to be considered moving

     r_static_mean: 12.0 # max brightness residual for static
     r_moving_mean: 25.0 # min brightness residual for moving

     k_static: 6         # frames in a row to confirm static
     k_moving: 2         # frames in a row to confirm moving
    ````

    - Decrease d_static_px / r_static_mean if you want stricter static detection.

    - Increase d_moving_px / r_moving_mean if you want more confident moving detection.

    - Increase k_static / k_moving if you want more stable (less flickering) decisions over time.
## 🙏 Acknowledgements

This project would not be possible without the excellent open-source work and datasets from the community:

- **Ultralytics YOLO11**  
  We build our detection and tracking pipeline on top of the Ultralytics YOLO ecosystem.

- **YOLO11l VisDrone checkpoint**  
  We use the VisDrone-finetuned YOLO11l model released by  
  [`erbayat/yolov11l-visdrone`](https://huggingface.co/erbayat/yolov11l-visdrone)  
  as our default detector for UAV / oblique-view scenarios.

- **VisDrone Dataset**  
  We rely on the VisDrone benchmark for training and evaluation of UAV detection scenarios:  
  P. Zhu *et al.*, “Vision Meets Drones: A Challenge,” *ECCV Workshops*, 2018.  
  Dataset homepage: https://github.com/VisDrone/VisDrone-Dataset

- **UAV Benchmark (UAVDT)**  
  We also make use of the UAVDT benchmark for UAV-based object detection and tracking:  
  D. Du *et al.*, “The Unmanned Aerial Vehicle Benchmark: Object Detection and Tracking,” ECCV 2018.  
  Benchmark homepage: https://sites.google.com/view/grli-uavdt/

If you use this repository in your research or projects, please also remember to cite and respect the
licenses / terms of use of the above models and datasets.
## ✅ TODO / Future Work

This repo is a practical first version. A few clear next steps are:

- [ ] **Camera-aware thresholds & speed scaling**  
  Pixel displacement thresholds for static/moving are currently hand-tuned.  
  Future work: use camera intrinsics (focal length, FOV) and approximate UAV altitude  
  to adapt these thresholds and improve speed accuracy.

- [ ] **Stronger feature extraction & matching for homography**  
  We now use ORB + BFMatcher for background feature matching.  
  Future work: try alternative feature / matcher combinations (e.g. SIFT or learned features)  
  to make camera motion estimation more stable across different textures and viewpoints.

- [ ] **Geometry-aware scale estimation (beyond fixed vehicle length)**  
  At the moment, speed scaling mainly uses a fixed “typical length” per class and the bbox long edge.  
  Future work: better account for the **viewing angle and vehicle pose** (e.g. how the long edge is foreshortened  
  along the road direction), and combine this with camera geometry to estimate a more accurate meters-per-pixel  
  for each track, rather than relying on a single rough length per class.



