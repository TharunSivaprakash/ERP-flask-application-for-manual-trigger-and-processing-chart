from flask import Flask, render_template, jsonify, request
import serial
import time
import logging
import csv
import os
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg for interactive window
import matplotlib.pyplot as plt
import mne
import numpy as np
from PIL import Image
import tempfile

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Hardcoded config
SERIAL_PORT = "COM5"
BAUD_RATE = 9600
PULSE_DURATION = 0.05
UPLOAD_FOLDER = "Uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs("exported_erps", exist_ok=True)

ALLOWED_EXTENSIONS = {"csv", "edf"}
ALLOWED_IMAGE_EXTENSIONS = {"jpeg", "jpg", "png"}

# Global state for sequence
schedule = []
current_repeat = 0
total_repeats = 1

# Store active Matplotlib figures to prevent garbage collection
active_figures = []

# Enable interactive mode
plt.ion()

def allowed_file(filename, extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in extensions

def send_pulse():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        ser.write(b"1")
        time.sleep(PULSE_DURATION)
        ser.close()
        return {"success": True, "message": "Pulse sent successfully"}
    except serial.SerialException as e:
        return {"success": False, "message": f"Serial error: {str(e)}"}
    except Exception as e:
        return {"success": False, "message": f"Unexpected error: {str(e)}"}

def close_active_figures():
    global active_figures
    for fig in active_figures:
        try:
            plt.close(fig)
        except:
            pass
    active_figures = []

def process_eeg(edf_path, signal_name, channels, time_of_interest, maskzoom, start, end, smoothing_ms=20):
    try:
        # Close previous figures to prevent multiple windows
        close_active_figures()

        raw = mne.io.read_raw_edf(edf_path, preload=True)
        available_channels = raw.ch_names
        invalid_channels = [ch for ch in channels if ch not in available_channels]
        if invalid_channels:
            return {"success": False, "message": f"Channels not found in EDF: {', '.join(invalid_channels)}"}

        events, event_id_all = mne.events_from_annotations(raw)
        if 'Patient Button' not in event_id_all:
            return {"success": False, "message": "'Patient Button' annotation missing"}

        event_id = {'PatientButton': event_id_all['Patient Button']}
        sfreq = raw.info['sfreq']
        shift_samples = int(0.070 * sfreq)
        events[:, 0] -= shift_samples

        tmin, tmax = -1.01, 2.4
        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                            baseline=(None, 0), preload=True)
        if len(epochs) == 0:
            return {"success": False, "message": "No epochs found after processing"}

        evoked = epochs.average(method='mean')

        kernel_size = int(smoothing_ms * sfreq / 1000)
        peak_info = []
        times_array = evoked.times

        # Plot in Matplotlib window
        fig = plt.figure(figsize=(10, 6))
        all_masked_values = []
        for channel in channels:
            evoked_channel = evoked.copy().pick([channel])
            data = evoked_channel.data[0]
            smoothed = np.convolve(data, np.ones(kernel_size)/kernel_size, mode='same')
            if maskzoom == 'mask' and start != end:
                masked = np.ma.masked_where((times_array > start) & (times_array < end), smoothed)
                valid_times = times_array[~masked.mask]
                valid_data = masked.compressed()
                if len(valid_times) > 0:
                    masked = np.interp(times_array, valid_times, valid_data)
                all_masked_values.append(masked)
                plt.plot(times_array, masked, label=f'{channel} (masked)')
            elif maskzoom == 'zoom':
                masked = np.ma.masked_where((times_array < start) | (times_array > end), smoothed)
                all_masked_values.append(masked)
                plt.plot(times_array, masked, label=f'{channel} (zoomed)')
            else:
                all_masked_values.append(smoothed)
                plt.plot(times_array, smoothed, label=f'{channel} (smoothed)')

            ch_name, peak_lat, peak_amp = evoked_channel.get_peak(mode='neg', return_amplitude=True)
            peak_info.append({
                'channel': channel,
                'latency': peak_lat,
                'amplitude': peak_amp
            })

        if not all_masked_values:
            return {"success": False, "message": "No valid data to plot"}

        max_y = max([np.max(m) for m in all_masked_values if np.any(np.isfinite(m))])
        plt.text(time_of_interest, max_y, signal_name, color='red', ha='center', va='bottom')
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (µV)")
        plt.title(f"ERP Visualization of {signal_name}")
        plt.legend()
        plt.grid(True)

        # Store figure to prevent garbage collection
        active_figures.append(fig)
        
        # Ensure the figure is drawn
        fig.canvas.draw()
        fig.canvas.flush_events()
        fig.canvas.manager.window.wm_title(f"ERP Visualization of {signal_name}")
        plt.show(block=True)  # Keep the plot window open


        # Export PNG/PDF
        filename_prefix = f"ERP_{signal_name}_{os.path.basename(edf_path).split('.')[0]}"
        png_path = f"exported_erps/{filename_prefix}.png"
        pdf_path = f"exported_erps/{filename_prefix}.pdf"
        export_fig = plt.figure(figsize=(10, 6))
        for channel in channels:
            evoked_channel = evoked.copy().pick([channel])
            data = evoked_channel.data[0]
            smoothed = np.convolve(data, np.ones(kernel_size)/kernel_size, mode='same')
            plt.plot(times_array, smoothed, label=f'{channel} (smoothed)')
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (µV)")
        plt.title(f"ERP Visualization of {signal_name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(png_path, dpi=300)
        plt.savefig(pdf_path)
        plt.close(export_fig)  # Close export figure only

        return {
            "success": True,
            "peak_info": peak_info,
            "exported_paths": [png_path, pdf_path]
        }
    except Exception as e:
        logging.error(f"EEG processing error: {str(e)}")
        return {"success": False, "message": f"EEG processing failed: {str(e)}"}

def get_edf_channels(edf_path):
    try:
        logging.debug(f"Attempting to read EDF file: {edf_path}")
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose='ERROR')
        channels = raw.ch_names
        logging.debug(f"Successfully read channels: {channels}")
        return {"success": True, "channels": channels}
    except FileNotFoundError:
        logging.error(f"EDF file not found: {edf_path}")
        return {"success": False, "message": f"EDF file not found: {edf_path}"}
    except Exception as e:
        logging.error(f"Error reading EDF channels: {str(e)}")
        return {"success": False, "message": f"Failed to read channels: {str(e)}"}

@app.route("/")
def index():
    static_folder = os.path.join(app.root_path, STATIC_FOLDER)
    images = [f for f in os.listdir(static_folder) if allowed_file(f, ALLOWED_IMAGE_EXTENSIONS)]
    return render_template("index.html", images=images)

@app.route("/get_images")
def get_images():
    static_folder = os.path.join(app.root_path, STATIC_FOLDER)
    images = [f for f in os.listdir(static_folder) if allowed_file(f, ALLOWED_IMAGE_EXTENSIONS)]
    return jsonify({"status": "success", "images": images})

@app.route("/upload_image", methods=["POST"])
def upload_image():
    if "image_file" not in request.files:
        logging.error("No image file uploaded")
        return jsonify({"status": "error", "message": "No image file uploaded"}), 400
    file = request.files["image_file"]
    if file.filename == "":
        logging.error("No image file selected")
        return jsonify({"status": "error", "message": "No image file selected"}), 400
    if file and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        filename = file.filename
        file_path = os.path.join(STATIC_FOLDER, filename)
        try:
            file.save(file_path)
            logging.debug(f"Image saved: {file_path}")
            return jsonify({"status": "success", "message": f"Image {filename} uploaded successfully"})
        except Exception as e:
            logging.error(f"Failed to save image: {str(e)}")
            return jsonify({"status": "error", "message": f"Failed to save image: {str(e)}"}), 500
    return jsonify({"status": "error", "message": "Invalid file type"}), 400

@app.route("/get_edf_channels", methods=["POST"])
def get_edf_channels_route():
    if "edf_file" not in request.files:
        logging.error("No EDF file uploaded")
        return jsonify({"status": "error", "message": "No EDF file uploaded"}), 400
    file = request.files["edf_file"]
    if file.filename == "" or not allowed_file(file.filename, {"edf"}):
        logging.error(f"Invalid or no EDF file selected: {file.filename}")
        return jsonify({"status": "error", "message": "Invalid or no EDF file selected"}), 400

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as temp:
            file.save(temp.name)
            temp_path = temp.name
            logging.debug(f"Saved temporary EDF file: {temp_path}")
        
        result = get_edf_channels(temp_path)
        os.unlink(temp_path)
        logging.debug(f"Deleted temporary file: {temp_path}")
        return jsonify({
            "status": "success" if result["success"] else "error",
            "channels": result.get("channels", []),
            "message": result.get("message", "")
        })
    except Exception as e:
        logging.error(f"Error in get_edf_channels_route: {str(e)}")
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return jsonify({"status": "error", "message": f"Failed to process EDF file: {str(e)}"}), 500

@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    global schedule
    if "csv_file" not in request.files:
        logging.error("No CSV file uploaded")
        return jsonify({"status": "error", "message": "No file uploaded"}), 400
    file = request.files["csv_file"]
    if file.filename == "":
        logging.error("No CSV file selected")
        return jsonify({"status": "error", "message": "No file selected"}), 400
    if file and allowed_file(file.filename, {"csv"}):
        filename = "triggers.csv"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        try:
            with open(file_path, "r") as f:
                reader = csv.DictReader(f)
                required_columns = ["image", "duration", "trigger", "repeat"]
                if not all(col in reader.fieldnames for col in required_columns):
                    missing = [col for col in required_columns if col not in reader.fieldnames]
                    return jsonify({"status": "error", "message": f"CSV missing columns: {', '.join(missing)}"}), 400
                static_folder = os.path.join(app.root_path, STATIC_FOLDER)
                cycle = []
                max_repeat = 0
                for row in reader:
                    image_path = os.path.join(static_folder, row["image"])
                    if not os.path.exists(image_path):
                        logging.error(f"Image not found: {row['image']}")
                        return jsonify({"status": "error", "message": f"Image not found: {row['image']}"}), 400
                    try:
                        repeat = int(row["repeat"])
                        if repeat < 1:
                            raise ValueError("Repeat must be at least 1")
                        max_repeat = max(max_repeat, repeat)
                        cycle.append({
                            "image": row["image"],
                            "duration": int(float(row["duration"])),
                            "trigger": int(row["trigger"])
                        })
                    except ValueError as e:
                        logging.error(f"Invalid CSV data: {str(e)}")
                        return jsonify({"status": "error", "message": f"Invalid data in CSV: {str(e)}"}), 400
                schedule = []
                for _ in range(max_repeat):
                    schedule.extend(cycle)
            logging.debug(f"CSV schedule loaded: {len(schedule)} items")
            return jsonify({"status": "success", "schedule": schedule})
        except Exception as e:
            logging.error(f"Failed to parse CSV: {str(e)}")
            return jsonify({"status": "error", "message": f"Failed to parse CSV: {str(e)}"}), 400
    return jsonify({"status": "error", "message": "Invalid file type"}), 400

@app.route("/process_eeg", methods=["POST"])
def process_eeg_route():
    if "edf_file" not in request.files:
        logging.error("No EDF file uploaded")
        return jsonify({"status": "error", "message": "No EDF file uploaded"}), 400
    file = request.files["edf_file"]
    if file.filename == "" or not allowed_file(file.filename, {"edf"}):
        logging.error(f"Invalid or no EDF file selected: {file.filename}")
        return jsonify({"status": "error", "message": "Invalid or no EDF file selected"}), 400

    data = request.form
    signal_name = data.get("signal_name", "P100")
    channels = data.getlist("channels")
    time_of_interest = float(data.get("time_of_interest", 100)) / 1000
    maskzoom = data.get("maskzoom", "mask")
    start = float(data.get("start", 0)) / 1000
    end = float(data.get("end", 0)) / 1000

    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as temp_file:
        file.save(temp_file.name)
        temp_file_path = temp_file.name

    try:
        result = process_eeg(temp_file_path, signal_name, channels, time_of_interest, maskzoom, start, end)
        return jsonify({
            "status": "success" if result["success"] else "error",
            "peak_info": result.get("peak_info"),
            "exported_paths": result.get("exported_paths"),
            "message": result.get("message")
        })
    except Exception as e:
        logging.error(f"Error in process_eeg_route: {str(e)}")
        return jsonify({"status": "error", "message": f"EEG processing error: {str(e)}"})
    finally:
        os.unlink(temp_file_path)

@app.route("/start_sequence")
def start_sequence():
    global schedule, current_repeat, total_repeats
    if not schedule:
        logging.error("No CSV schedule loaded")
        return jsonify({"status": "error", "message": "No CSV schedule loaded"}), 400
    static_folder = os.path.join(app.root_path, STATIC_FOLDER)
    for item in schedule:
        if not os.path.exists(os.path.join(static_folder, item["image"])):
            logging.error(f"Image not found for sequence: {item['image']}")
            return jsonify({"status": "error", "message": f"Image not found: {item['image']}"}), 400
    current_repeat = 1
    total_repeats = 1
    logging.debug(f"Starting sequence with {len(schedule)} items")
    return jsonify({
        "status": "success",
        "message": "Sequence started",
        "current_repeat": current_repeat,
        "total_repeats": total_repeats
    })

@app.route("/manual_trigger", methods=["POST"])
def manual_trigger():
    global schedule, current_repeat, total_repeats
    data = request.get_json()
    try:
        selections = data.get("selections", [])
        if len(selections) < 2:
            logging.error("Less than two images selected for manual trigger")
            return jsonify({"status": "error", "message": "Select at least two images"}), 400
        static_folder = os.path.join(app.root_path, STATIC_FOLDER)
        cycle = []
        max_repeat = 0
        for sel in selections:
            image_path = os.path.join(static_folder, sel["image"])
            if not os.path.exists(image_path):
                logging.error(f"Image not found for manual trigger: {sel['image']}")
                return jsonify({"status": "error", "message": f"Image not found: {sel['image']}"}), 400
            try:
                repeat = int(sel["repeat"])
                if repeat < 1:
                    raise ValueError("Repeat must be at least 1")
                max_repeat = max(max_repeat, repeat)
                cycle.append({
                    "image": sel["image"],
                    "duration": int(float(sel["duration"])),
                    "trigger": 1 if sel["trigger"] else 0
                })
            except ValueError as e:
                logging.error(f"Invalid manual trigger data: {str(e)}")
                return jsonify({"status": "error", "message": f"Invalid data: {str(e)}"}), 400
        schedule = []
        for _ in range(max_repeat):
            schedule.extend(cycle)
        current_repeat = 1
        total_repeats = 1
        logging.debug(f"Manual trigger schedule: {len(schedule)} items")
        return jsonify({
            "status": "success",
            "message": "Manual sequence started",
            "current_repeat": current_repeat,
            "total_repeats": total_repeats
        })
    except Exception as e:
        logging.error(f"Error in manual_trigger: {str(e)}")
        return jsonify({"status": "error", "message": f"Failed to process selections: {str(e)}"}), 400

@app.route("/trigger")
def trigger():
    result = send_pulse()
    if result["success"]:
        return jsonify({"status": "success", "message": result["message"]})
    else:
        return jsonify({"status": "error", "message": result["message"]}), 500

@app.route("/fullscreen")
def fullscreen():
    global schedule, current_repeat
    repeat = int(request.args.get("repeat", 1))
    total = int(request.args.get("total", 1))
    if not schedule:
        logging.error("No schedule loaded for fullscreen")
        return jsonify({"status": "error", "message": "No schedule loaded"}), 400
    static_folder = os.path.join(app.root_path, STATIC_FOLDER)
    for item in schedule:
        if not os.path.exists(os.path.join(static_folder, item["image"])):
            logging.error(f"Image not found in fullscreen: {item['image']}")
            return jsonify({"status": "error", "message": f"Image not found: {item['image']}"}), 400
    logging.debug(f"Rendering fullscreen with {len(schedule)} items")
    return render_template("fullscreen.html", schedule=schedule, current_repeat=repeat, total_repeats=total)

if __name__ == "__main__":
    import webbrowser
    import time
    port = 5000
    url = f"http://localhost:{port}"
    time.sleep(1)  # Wait for server to start
    webbrowser.open_new(url)  # Open single tab
    app.run(host="0.0.0.0", port=port, debug=False)  # Disable debug