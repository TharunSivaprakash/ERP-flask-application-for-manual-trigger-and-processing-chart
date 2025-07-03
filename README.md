# EEG Trigger Experiment

This project is a **Flask-based web application** for conducting EEG experiments with synchronized image presentation and hardware triggering.  

It displays a sequence of images in fullscreen mode while triggering a relay on a **NodeMCU** via serial communication. The application also processes EEG data and visualizes event-related potentials.

---

##  Features

- **Image Sequence Presentation**
  - Displays a sequence of images (`stimulus4.png`, `stimulus5.png`) defined in a `triggers.csv` file.
  - Each image is shown for a specified duration (e.g., 3 seconds).
  - The sequence repeats for a configurable number of cycles (default: 60).

- **Relay Triggering**
  - Sends trigger pulses to a **NodeMCU** connected on **COM5** for each image presentation.

- **EEG Data Processing**
  - Upload and process EEG data (`EDF` files) using **MNE-Python**.
  - Visualize ERPs in a **Matplotlib** window.
  - Export plots as **PNG** and **PDF**.

- **Manual Trigger Mode**
  - Select images, durations, and repeats manually through the web interface.

- **Fullscreen Display**
  - Smooth image transitions with fade effects (0.5s).
  - Optimized for single-tab operation.

---

## 📂 Prerequisites

- **Python 3.8+**
- **NodeMCU** connected to **COM5**
  - Flash the provided `nodemcu_eeg_trigger.ino` firmware.
- **Images**
  - `stimulus4.png`
  - `stimulus5.png` (place in the `static/` folder)
- **CSV File**
  - `triggers.csv` (place in the `uploads/` folder)

---

##  Dependencies

Install required Python packages:
pip install Flask pyserial mne numpy matplotlib Pillow


```bash
pip install Flask pyserial mne numpy matplotlib Pillow
