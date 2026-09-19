# FRAS — Facial Recognition Attendance System

FRAS is a classroom attendance tool built using Python's `face_recognition`
library. Each student registers their face once; after that, whenever they
appear in front of the camera, the system automatically marks their
attendance, stores it in a CSV file, and generates a PDF report for the
instructor.

## Features

- **Student registration** — capture up to 10 face images per student via webcam
  (spacebar or automatic capture) and store them under `dataset/<student_name>/`.
- **Live attendance** — detects and recognizes faces from a webcam feed and logs
  attendance to `attendance_log.csv`, with a 75-minute cooldown per student to
  prevent duplicate entries in the same session.
- **PDF reports** — summarizes attendance by date into a formatted PDF report.

## Requirements

- Python 3.9+
- A webcam
- `cmake` and a C++ compiler (required to build `dlib`, a dependency of
  `face_recognition`)

## Installation

```bash
git clone https://github.com/Abhilash-reddyS/FRAS.git
cd FRAS
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> `face_recognition` depends on `dlib`, which is compiled from source on most
> platforms. If installation fails, make sure `cmake` is installed first
> (`brew install cmake` on macOS, `apt install cmake` on Debian/Ubuntu).

## Usage

```bash
python3 main.py
```

You'll see a menu:

```
===== Facial Recognition Attendance System =====
1. Register New Student
2. Start Attendance System
3. Generate Attendance Report
4. Exit
```

1. **Register New Student** — enter a name, then press `SPACEBAR` to capture
   each image (`q` to stop early).
2. **Start Attendance System** — opens the webcam and marks attendance for any
   recognized student; press `q` to quit.
3. **Generate Attendance Report** — creates a timestamped PDF summarizing
   attendance by date.

## Project structure

```
main.py                  Entry point / CLI menu
student_registration.py  Webcam capture and dataset creation
attendance_system.py     Face recognition and attendance logging
report_generator.py      PDF report generation from attendance_log.csv
```

## Data generated at runtime (not committed)

- `dataset/` — captured student face images
- `attendance_log.csv` — attendance records
- `attendance_report_*.pdf` — generated reports

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
