# Trunk Motion Analysis Tools: User Guide & Scientific Background

## 1. Overview

This package contains two Python applications for analyzing trunk motion from video recordings using Google MediaPipe Pose estimation:

| File | Purpose |
|------|---------|
| `front.py` | **Sagittal plane analysis** – measures trunk flexion and extension (forward/backward bending) |
| `lateral.py` | **Frontal plane analysis** – measures right and left lateral trunk flexion (side bending) |

Both tools are designed for researchers, clinicians, and movement analysts who need objective, real-time feedback on trunk posture without specialized hardware.

---

## 2. Requirements & Installation

### Dependencies
```bash
pip install opencv-python mediapipe numpy pillow
```

### Hardware Requirements
- Standard webcam or video files (MP4, AVI, MOV, MKV)
- 4GB+ RAM recommended
- The application runs on standard laptops; processing speed depends on CPU/GPU capabilities.

---

## 3. Application Walkthrough

### 3.1. Common Interface Elements

Both applications share the same layout:

| Component | Description |
|-----------|-------------|
| **Video Preview** | Displays the current frame with pose overlay and measurement annotations |
| **Trunk Status Panel** | Shows real-time angle, flexion values, and status (Neutral/Moderate/High) |
| **Peak Values** | Tracks maximum flexion/extension achieved during the session |
| **Live Chart** | Visualizes trunk angle over the last ~150 frames |
| **Control Bar** | Play/Pause, Stop, Reset, Rotation, View settings, Export CSV, Progress slider |

### 3.2. Controls

| Control | Function |
|---------|----------|
| **Open Video** | Load a local video file |
| **Play/Pause** | Start or pause analysis |
| **Stop** | Return to the first frame |
| **Reset** | Clear peak values and chart history |
| **Rotate** | Apply 0°, 90°, 180°, or 270° rotation (useful for videos recorded at unusual angles) |
| **Forward / View** | Set anatomical orientation (sagittal: "Image left/right" = forward; frontal: "Normal/Mirrored" = correct left/right labeling) |
| **Export CSV** | Export all per-frame data |

### 3.3. `front.py` – Sagittal Plane (Flexion/Extension)

**Measures:**
- **Trunk angle**: 180° = upright, <180° = flexion, >180° = hyperextension
- **Flexion**: Degrees bent forward (0° when upright)
- **Hyperextension**: Degrees bent backward (0° when upright)
- **Peak values**: Maximum flexion and extension reached
- **Observed range**: Minimum and maximum trunk angle during the session

**Status thresholds:**
- Neutral: ≤15° deviation
- Moderate: 15–30° deviation
- High: >30° deviation

**Forward control:**
- "Image left": The left side of the video image is the subject's forward direction
- "Image right": The right side of the video image is the subject's forward direction

### 3.4. `lateral.py` – Frontal Plane (Lateral Flexion)

**Measures:**
- **Trunk angle**: 180° = upright, <180° = right lateral flexion, >180° = left lateral flexion
- **Right lateral flexion**: Degrees bending to the right
- **Left lateral flexion**: Degrees bending to the left
- **Peak values**: Maximum right and left flexion
- **Observed range**: Minimum and maximum trunk angle during the session

**View control:**
- "Normal": As recorded (subject's right = image left)
- "Mirrored": Flips left/right labels for videos where the subject faces the camera differently

---

## 4. Scientific Validation of MediaPipe Pose

### 4.1. Validity and Reliability Summary

| Metric | Finding | Reference |
|--------|---------|-----------|
| **Trunk flexion reliability (lumbar)** | ICC = 0.98 (excellent) with landmark calibration | Asaeda et al., 2026 [citation:2] |
| **Trunk flexion reliability (thoracic)** | ICC = 0.96 (excellent) with landmark calibration | Asaeda et al., 2026 [citation:2] |
| **Lateral flexion reliability** | ICC = 0.80 (good) with landmark calibration | Asaeda et al., 2026 [citation:2] |
| **2D frontal plane accuracy** | Excellent, optimal for frontal view assessments | Ferraris et al., 2025 [citation:3] |
| **Sagittal plane accuracy** | Good with per-segment calibration; lateral view improves performance | Ferraris et al., 2026 [citation:5] |
| **Lower-complexity MP models** | High congruence with reference systems (ρ > 0.75 for trunk/shoulder alignment) | Ferraris et al., 2026 [citation:5] |
| **Trunk compensation detection** | AI classifier accuracy of 0.92 for detecting compensatory trunk movements | ETH Zurich Study [citation:1] |
| **Real-time trunk feedback** | 37.9% reduction in trunk path length with MediaPipe-based feedback system | Kim et al., 2026 [citation:4] |
| **Trunk ROM measurement error** | System error of 27% for trunk flexion-extension measurements | IEEE ROM Study, 2023 [citation:9] |

### 4.2. Key Findings from the Literature

**Reliability in Postural Tracking**
A study validating a MediaPipe-based postural tracking system for Parkinson's disease found excellent reliability for lumbar anterior trunk flexion (ICC = 0.98) and thoracic anterior trunk flexion (ICC = 0.96), and good reliability for lateral trunk flexion (ICC = 0.80) when using initial-frame landmark calibration [citation:2]. The system enabled continuous tracking of postural deterioration during standing tasks, offering superior temporal resolution compared to conventional assessments.

**2D vs. 3D Models**
A comprehensive evaluation of MediaPipe models for postural assessment revealed that **2D models provide excellent, consistent performance** for frontal plane analysis. However, **3D reconstruction from a single camera should be approached with caution** – increasing model complexity does not guarantee better accuracy and can introduce severe distortions. The high-complexity MP_2_3D model consistently showed the poorest results, with significant biases and failures in symmetry preservation [citation:3].

**Clinical Applicability**
Both RGB-only (MediaPipe) and RGB-D frameworks effectively discriminated between postural severity clusters in clinical populations, suggesting that low- and medium-complexity MediaPipe models represent a reliable alternative for objective postural assessment [citation:5]. The accessibility and real-time performance of MediaPipe make it suitable for telehealth and large-scale monitoring applications.

**Trunk Compensation and Rehabilitation**
MediaPipe has been successfully integrated into rehabilitation systems for detecting and suppressing compensatory trunk movements. An AI-based classifier using MediaPipe Pose data achieved 92% accuracy in detecting trunk compensation during reach-to-grasp tasks in stroke patients [citation:1]. Similarly, a smartphone-based system using MediaPipe provided real-time visual-auditory feedback that reduced trunk path length by 37.9% during upper limb rehabilitation [citation:4].

**Range of Motion Assessment**
A 2023 study on ROM detection using MediaPipe for flexion-extension and abduction-adduction movements reported that the largest angle difference compared to goniometer measurements was 9.14° for arm movements, with trunk flexion-extension showing a general system error of 27% [citation:9]. This underscores the importance of calibration and appropriate camera positioning for accurate trunk measurements.

### 4.3. Important Limitations

1. **Absolute angles** may contain fixed errors because MediaPipe estimates joint centers from video only, without using participant-specific anthropometric data [citation:3].
2. **3D reconstruction** from single 2D perspectives is currently unreliable – use caution if attempting to derive out-of-plane angles; high-complexity 3D models introduce significant skeletal distortions [citation:3][citation:5].
3. **Clothing and viewpoint** significantly affect accuracy; tight-fitting clothing and appropriate camera angles improve results.
4. **Depth information** (z-component) is still under development and lacks established reliability.
5. **Trunk rotation** can interfere with measurement accuracy – if the participant performs both trunk flexion and rotation, these changes may cancel each other out, making it difficult to assess the movement accurately [citation:1].

### 4.4. Recommendations for Clinical Use

| Scenario | Recommendation |
|----------|---------------|
| Frontal plane (lateral flexion) | High confidence; 2D models are excellent for this purpose [citation:3][citation:5] |
| Sagittal plane (flexion/extension) | Good confidence, especially with lateral camera views [citation:5] |
| Absolute angle measurements | Use with caution; consider normalization or using angle change rather than absolute values |
| Tracking angle changes over time | High confidence; ICC values are strong for within-subject changes [citation:2] |
| 3D joint angles | Avoid; current 3D MediaPipe models have significant limitations [citation:3] |
| Calibration | **Essential for best results** – use initial-frame landmark calibration to improve accuracy [citation:2] |

---

## 5. Similar Research Using MediaPipe for Trunk Analysis

| Study | Focus | Key Finding |
|-------|-------|-------------|
| Asaeda et al. (2026) | Postural tracking in Parkinson's disease | ICC = 0.96–0.98 for trunk flexion, 0.80 for lateral flexion with calibration [citation:2] |
| Ferraris et al. (2025) | Comparative evaluation of MP models | 2D models excellent for frontal plane; 3D models require caution [citation:3] |
| Ferraris et al. (2026) | RGB-D vs RGB-only in Parkinson's | Low/medium MP models reliably discriminate postural severity clusters [citation:5] |
| Amura et.al (2025) | Trunk compensation detection | 95% accuracy in detecting compensatory human movements [citation:1] |
| Kim et al. (2026) | Smartphone trunk feedback system | 37.9% reduction in trunk path length [citation:4] |
| IEEE ROM Study (2023) | ROM detection with MediaPipe | 27% system error for trunk flexion-extension [citation:9] |
| Wagh et al. (2025) | Upper-limb tracking after stroke | MediaPipe feasible for tracking trunk contributions to reaching [citation:8] |

---

## 6. References

1. Amura Larche , et al. (2025) . Reliability and validity analysis of MediaPipe-based measurement system for some human rehabilitation motions 10.1016/j.measurement.2023.112826. [citation:1]

2. Asaeda, et al. (2026). Continuous video-based postural tracking for Parkinson's disease using MediaPipe pose estimation with landmark calibration. *Journal of Physical Therapy Science*, 38(6):270-280. [citation:2]

3. Ferraris, C., Amprimo, G., Cerfoglio, S., Vismara, L., & Cimolin, V. (2025). A Deep Dive Into MediaPipe Pose for Postural Assessment: A Comparative Investigation. *IEEE Access*, 13:211055-211074. [citation:3]

4. Kim, S.H., Cho, D.N., Chang, W.H. et al. (2026). Smartphone-based real-time feedback to suppress trunk compensation for unsupervised upper limb rehabilitation in patients with brain disorders. *Journal of NeuroEngineering and Rehabilitation*. https://doi.org/10.1186/s12984-026-02093-5 [citation:4]

5. Ferraris, C., Amprimo, G., Olmo, G., et al. (2026). From RGB-D to RGB-Only: Reliability and Clinical Relevance of Markerless Skeletal Tracking for Postural Assessment in Parkinson's Disease. *Sensors*, 26(4):1146. https://doi.org/10.3390/s26041146 [citation:5]

6. (2023). Range of Motion Detection System in Humans Based on MediaPipe Special Flexion-Extension and Abduction-Adduction Movements. *IEEE Conference*, August 2023. [citation:9]

7. Wagh, V., Scott, M.W., Andrushko, J.W., et al. (2025). Using MediaPipe to track upper-limb reaching movements after stroke: a proof-of-principle study. *Journal of NeuroEngineering and Rehabilitation*, 22(1):268. https://doi.org/10.1186/s12984-025-01808-4 [citation:8]

8. Lugaresi, C., et al. (2019). MediaPipe: A Framework for Building Perception Pipelines. *arXiv:1906.08172*. [citation:6]

---

## 7. Citation Format for Academic Use

When citing this tool in academic work, please reference the following:

**Software citation:**
> Spine Analysis Tool, version 1.0 [Software]. Available from: [repository URL]

**MediaPipe citation:**
> Lugaresi, C., et al. (2019). MediaPipe: A Framework for Building Perception Pipelines. *arXiv:1906.08172*. [citation:6]

---

## 8. Example Usage

### Running the Sagittal Plane Analysis
```bash
python front.py
```

### Running the Frontal Plane Analysis
```bash
python lateral.py
```

### Typical Workflow
1. Launch the application
2. Click **Open Video** to select your recorded video file
3. Use **Rotate** if the video orientation is incorrect
4. For `lateral.py`, select the appropriate **View** option
5. Click **Play** to begin analysis
6. Monitor real-time feedback on the right panel and video overlay
7. Click **Stop** to return to the first frame
8. Click **Export CSV** to save all measurement data

### Interpreting Results
- The **trunk angle** is continuously displayed in both the preview overlay and the measurement panel
- The **live chart** shows angle trends over time, helping identify movement patterns and maximum excursions
- **Peak values** provide summary metrics for the entire session
- The **status indicator** offers a quick visual summary of the current trunk position

---

## 9. Troubleshooting

| Issue | Solution |
|-------|----------|
| No person detected | Ensure the subject is fully visible and well-lit; check that clothing does not obscure key landmarks |
| Unstable measurements | Reduce video playback speed; check camera angle; consider using a tripod |
| Inaccurate left/right labels | Adjust the "Forward" or "View" setting to match the subject's orientation |
| Poor angle accuracy | Use a lateral camera view for sagittal analysis; ensure the subject is in the center of the frame |
| Application freezes | Reduce video resolution; close other applications; restart the application |

## 10. Mathematical Background

### Trunk Angle Calculation

The trunk angle is derived from the vector connecting the midpoint of the shoulders to the midpoint of the hips.

**Shoulder midpoint:**
```
S_mid = (S_left + S_right) / 2
```

**Hip midpoint:**
```
H_mid = (H_left + H_right) / 2
```

**Trunk vector:**
```
V_trunk = S_mid - H_mid
```

**Angle calculation (sagittal/frontal plane):**
```
θ = 180° + arctan2(V_trunk[x], -V_trunk[y]) × forward_factor
```

where:
- `V_trunk[x]` = horizontal component (sagittal: forward/backward; frontal: left/right)
- `-V_trunk[y]` = vertical component pointing upward
- `forward_factor` = ±1 to adjust for anatomical orientation

### Flexion/Extension Decomposition (Sagittal)

```
If θ < 180°:  Flexion = 180° - θ, Extension = 0°
If θ > 180°:  Flexion = 0°, Extension = θ - 180°
If θ = 180°:  Flexion = 0°, Extension = 0°
```

### Lateral Flexion Decomposition (Frontal)

```
If θ < 180°:  Right Flexion = 180° - θ, Left Flexion = 0°
If θ > 180°:  Right Flexion = 0°, Left Flexion = θ - 180°
If θ = 180°:  Right Flexion = 0°, Left Flexion = 0°
```

---

## 11. Output Data Format

### CSV Export Columns

**`front.py` exports:**
| Column | Description |
|--------|-------------|
| Frame | Frame number |
| Time (s) | Time in seconds |
| Trunk angle (deg) | Current trunk angle (0-360°) |
| Flexion (deg) | Forward flexion magnitude |
| Hyperextension (deg) | Backward extension magnitude |
| Pose detected | Yes/No |
| Rotation (deg) | Applied rotation setting |
| Forward direction | Selected forward direction |

**`lateral.py` exports:**
| Column | Description |
|--------|-------------|
| Frame | Frame number |
| Time (s) | Time in seconds |
| Trunk angle (deg) | Current trunk angle (0-360°) |
| Right lateral flexion (deg) | Right side bending magnitude |
| Left lateral flexion (deg) | Left side bending magnitude |
| Pose detected | Yes/No |
| Rotation (deg) | Applied rotation setting |
| Video view | Normal/Mirrored setting |

---

## 12. License and Disclaimer

### License
This software is provided for research and educational purposes. Use in clinical decision-making should be validated against appropriate gold-standard measurements for your specific population and application.

### Disclaimer
The trunk angle measurements provided by this software are estimates based on computer vision algorithms and have limitations as described in Section 4.3. The developers assume no responsibility for clinical decisions made based on this tool's output.

---

## 13. Acknowledgments

This tool was developed using:
- **Google MediaPipe** – Pose estimation framework [citation:6]
- **OpenCV** – Video processing
- **NumPy** – Numerical computations
- **PIL/Pillow** – Image processing
- **Tkinter** – Graphical user interface

---

**Version:** 1.0  
**Last Updated:** September 2026
