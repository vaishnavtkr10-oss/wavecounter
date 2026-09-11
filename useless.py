import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from collections import deque
import time


# ============================================================
# 🌊 SEA WAVE COUNTER
# ============================================================

WINDOW_NAME = "SEA WAVE COUNTER"


# ============================================================
# SETTINGS
# ============================================================

# Minimum seconds between two counted waves
MIN_WAVE_GAP = 2.5

# Number of frames used for smoothing
SMOOTHING_FRAMES = 7

# Number of recent measurements
HISTORY_SIZE = 90

# Wave detection sensitivity
MOTION_THRESHOLD = 0.018
FOAM_THRESHOLD = 0.025

# How much stronger a wave must be than its recent background
PEAK_MULTIPLIER = 1.25

# Scan zone position inside selected sea
# 0.0 = top of sea
# 1.0 = bottom of sea
SCAN_POSITION = 0.72

# Height of scanning zone
SCAN_HEIGHT_RATIO = 0.10

# Number of horizontal bands
NUMBER_OF_BANDS = 5


# ============================================================
# VIDEO SELECTION
# ============================================================

def select_video():

    root = tk.Tk()
    root.withdraw()

    path = filedialog.askopenfilename(
        title="Select Sea Video",
        filetypes=[
            ("MP4 Video", "*.mp4"),
            ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
            ("All Files", "*.*")
        ]
    )

    root.destroy()

    return path


# ============================================================
# TEXT
# ============================================================

def put_text(
    frame,
    message,
    position,
    size=0.7,
    thickness=2
):

    cv2.putText(
        frame,
        message,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# CREATE WATER MASK
# ============================================================

def water_mask(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # --------------------------------------------------------
    # Detect blue / cyan / turquoise water
    # --------------------------------------------------------

    blue_water = (
        (h >= 75) &
        (h <= 115) &
        (s >= 25) &
        (v >= 45)
    )

    cyan_water = (
        (h >= 50) &
        (h < 75) &
        (s >= 25) &
        (v >= 45)
    )

    green_water = (
        (h >= 35) &
        (h < 50) &
        (s >= 25) &
        (v >= 50)
    )

    mask = (
        blue_water |
        cyan_water |
        green_water
    )

    mask = mask.astype(np.uint8) * 255

    # Remove tiny noise
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


# ============================================================
# FOAM DETECTION
# ============================================================

def foam_amount(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    saturation = hsv[:, :, 1]
    brightness = hsv[:, :, 2]

    # White / bright foam
    foam = (
        (brightness > 155) &
        (saturation < 170)
    )

    foam = foam.astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)

    foam = cv2.morphologyEx(
        foam,
        cv2.MORPH_OPEN,
        kernel
    )

    foam = cv2.morphologyEx(
        foam,
        cv2.MORPH_CLOSE,
        kernel
    )

    amount = np.count_nonzero(foam) / foam.size

    return amount, foam


# ============================================================
# MOTION DETECTION
# ============================================================

def motion_amount(previous, current):

    if previous is None:
        return 0.0

    previous_gray = cv2.cvtColor(
        previous,
        cv2.COLOR_BGR2GRAY
    )

    current_gray = cv2.cvtColor(
        current,
        cv2.COLOR_BGR2GRAY
    )

    # Small blur removes camera/noise fluctuations
    previous_gray = cv2.GaussianBlur(
        previous_gray,
        (7, 7),
        0
    )

    current_gray = cv2.GaussianBlur(
        current_gray,
        (7, 7),
        0
    )

    difference = cv2.absdiff(
        previous_gray,
        current_gray
    )

    # Adaptive threshold
    threshold_value = 18

    motion = difference > threshold_value

    amount = np.count_nonzero(motion) / motion.size

    return amount


# ============================================================
# COMBINED WAVE SIGNAL
# ============================================================

def wave_signal(
    current_frame,
    previous_frame
):

    foam, foam_mask = foam_amount(
        current_frame
    )

    motion = motion_amount(
        previous_frame,
        current_frame
    )

    # Motion is slightly more important than foam.
    signal = (
        motion * 0.60 +
        foam * 0.40
    )

    return signal, motion, foam, foam_mask


# ============================================================
# DRAW WAVE DETECTION LINE
# ============================================================

def draw_detection_zone(
    frame,
    x,
    y,
    w,
    h
):

    scan_y = int(
        y + h * SCAN_POSITION
    )

    scan_height = max(
        10,
        int(h * SCAN_HEIGHT_RATIO)
    )

    scan_top = max(
        y,
        scan_y - scan_height // 2
    )

    scan_bottom = min(
        y + h,
        scan_y + scan_height // 2
    )

    # Outer sea rectangle
    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        (255, 255, 0),
        2
    )

    # Wave scanning zone
    cv2.rectangle(
        frame,
        (x, scan_top),
        (x + w, scan_bottom),
        (0, 255, 0),
        2
    )

    # Main wave detection line
    cv2.line(
        frame,
        (x, scan_y),
        (x + w, scan_y),
        (0, 0, 255),
        3
    )

    put_text(
        frame,
        "WAVE DETECTION LINE",
        (x + 10, scan_top - 10),
        0.55,
        2
    )

    return scan_top, scan_bottom, scan_y


# ============================================================
# RESET DETECTOR
# ============================================================

def reset_detector():

    return {
        "wave_count": 0,

        "signal_history": deque(
            maxlen=HISTORY_SIZE
        ),

        "smooth_history": deque(
            maxlen=SMOOTHING_FRAMES
        ),

        "previous_frame": None,

        "last_wave_time": -999999,

        "wave_active": False,

        "wave_start_time": 0,

        "status": "SCANNING..."
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("🌊 SEA WAVE COUNTER")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # SELECT VIDEO
    # --------------------------------------------------------

    print("📁 Select your sea video...")

    video_path = select_video()

    if not video_path:

        print("❌ No video selected.")
        return

    print()
    print("✅ Video selected:")
    print(video_path)
    print()

    # --------------------------------------------------------
    # OPEN VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        print("❌ Could not open video.")
        return

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    duration = total_frames / fps

    print(
        f"Resolution : {width} x {height}"
    )

    print(
        f"FPS        : {fps:.2f}"
    )

    print(
        f"Duration   : {duration:.1f} seconds"
    )

    print()

    # --------------------------------------------------------
    # READ FIRST FRAME
    # --------------------------------------------------------

    ret, first_frame = cap.read()

    if not ret:

        print("❌ Could not read first frame.")

        cap.release()

        return

    # --------------------------------------------------------
    # SELECT SEA
    # --------------------------------------------------------

    print("=" * 70)
    print("SELECT THE SEA")
    print("=" * 70)
    print()

    print(
        "Draw a rectangle around the WATER."
    )

    print()

    print(
        "Try to exclude:"
    )

    print(
        "  ❌ Sky"
    )

    print(
        "  ❌ Beach"
    )

    print(
        "  ❌ Buildings"
    )

    print(
        "  ❌ Large objects"
    )

    print()

    print(
        "Press ENTER after selecting."
    )

    print()

    roi = cv2.selectROI(
        "SELECT SEA - Press ENTER",
        first_frame,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyWindow(
        "SELECT SEA - Press ENTER"
    )

    x, y, w, h = roi

    if w == 0 or h == 0:

        print("❌ No sea selected.")

        cap.release()

        return

    print()
    print("✅ Sea selected.")
    print()

    # --------------------------------------------------------
    # DETECTION ZONE
    # --------------------------------------------------------

    scan_top, scan_bottom, scan_y = (
        draw_detection_zone(
            first_frame,
            x,
            y,
            w,
            h
        )
    )

    print(
        f"🌊 Wave line Y = {scan_y}"
    )

    print()

    # --------------------------------------------------------
    # RESET VIDEO
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        0
    )

    # --------------------------------------------------------
    # DETECTOR
    # --------------------------------------------------------

    detector = reset_detector()

    frame_number = 0

    paused = False

    delay = max(
        1,
        int(1000 / fps)
    )

    current_signal = 0
    current_motion = 0
    current_foam = 0

    # --------------------------------------------------------
    # VIDEO LOOP
    # --------------------------------------------------------

    while True:

        if not paused:

            ret, frame = cap.read()

            if not ret:

                break

            frame_number += 1

            current_time = (
                frame_number / fps
            )

            # ------------------------------------------------
            # GET SEA
            # ------------------------------------------------

            sea = frame[
                y:y + h,
                x:x + w
            ]

            if sea.size == 0:

                continue

            # ------------------------------------------------
            # GET WAVE SCANNING ZONE
            # ------------------------------------------------

            scan = frame[
                scan_top:scan_bottom,
                x:x + w
            ]

            if scan.size == 0:

                continue

            # ------------------------------------------------
            # DETECT WAVE SIGNAL
            # ------------------------------------------------

            previous_scan = None

            if detector["previous_frame"] is not None:

                previous_scan = detector[
                    "previous_frame"
                ]

            signal, motion, foam, foam_mask = (
                wave_signal(
                    scan,
                    previous_scan
                )
            )

            current_signal = signal

            current_motion = motion

            current_foam = foam

            # Save current frame
            detector["previous_frame"] = scan.copy()

            # ------------------------------------------------
            # SMOOTH SIGNAL
            # ------------------------------------------------

            detector[
                "smooth_history"
            ].append(signal)

            smooth_signal = np.mean(
                detector[
                    "smooth_history"
                ]
            )

            detector[
                "signal_history"
            ].append(
                smooth_signal
            )

            history = np.array(
                detector[
                    "signal_history"
                ],
                dtype=np.float32
            )

            # ------------------------------------------------
            # WAVE DETECTION
            # ------------------------------------------------

            if len(history) >= 15:

                recent = history[-15:]

                baseline = np.median(
                    recent[:-3]
                )

                peak = np.max(
                    recent[-3:]
                )

                # Dynamic threshold
                dynamic_threshold = max(
                    MOTION_THRESHOLD,
                    baseline * PEAK_MULTIPLIER
                )

                # Wave signal must be strong
                strong_signal = (
                    peak >
                    dynamic_threshold
                )

                # Need either movement or foam
                physical_wave = (
                    motion >= MOTION_THRESHOLD
                    or
                    foam >= FOAM_THRESHOLD
                )

                # ------------------------------------------------
                # START WAVE
                # ------------------------------------------------

                if (
                    strong_signal
                    and
                    physical_wave
                    and
                    not detector["wave_active"]
                    and
                    current_time -
                    detector["last_wave_time"]
                    >= MIN_WAVE_GAP
                ):

                    detector[
                        "wave_active"
                    ] = True

                    detector[
                        "wave_start_time"
                    ] = current_time

                # ------------------------------------------------
                # CONFIRM WAVE
                # ------------------------------------------------

                if detector["wave_active"]:

                    wave_duration = (
                        current_time -
                        detector[
                            "wave_start_time"
                        ]
                    )

                    # Wave must remain visible briefly
                    if (
                        wave_duration >= 0.25
                        and
                        (
                            motion >=
                            MOTION_THRESHOLD
                            or
                            foam >=
                            FOAM_THRESHOLD
                        )
                    ):

                        detector[
                            "wave_count"
                        ] += 1

                        detector[
                            "last_wave_time"
                        ] = current_time

                        detector[
                            "wave_active"
                        ] = False

                        detector[
                            "status"
                        ] = (
                            "🌊 WAVE DETECTED!"
                        )

                        print(
                            f"🌊 Wave #"
                            f"{detector['wave_count']} "
                            f"at "
                            f"{current_time:.1f}s"
                        )

                # ------------------------------------------------
                # RESET WAVE STATE
                # ------------------------------------------------

                if detector["wave_active"]:

                    if (
                        current_time -
                        detector[
                            "wave_start_time"
                        ]
                        > 2.0
                    ):

                        detector[
                            "wave_active"
                        ] = False

            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            if not detector["wave_active"]:

                detector[
                    "status"
                ] = "SCANNING..."

            # ------------------------------------------------
            # DRAW SEA AREA
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 255, 0),
                2
            )

            # ------------------------------------------------
            # DRAW SCAN ZONE
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x, scan_top),
                (x + w, scan_bottom),
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # DRAW WAVE LINE
            # ------------------------------------------------

            cv2.line(
                frame,
                (x, scan_y),
                (x + w, scan_y),
                (0, 0, 255),
                3
            )

            put_text(
                frame,
                "WAVE DETECTION LINE",
                (x + 10, scan_top - 10),
                0.55,
                2
            )

            # ------------------------------------------------
            # TOP INFORMATION PANEL
            # ------------------------------------------------

            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (0, 0),
                (width, 135),
                (0, 0, 0),
                -1
            )

            frame = cv2.addWeighted(
                overlay,
                0.65,
                frame,
                0.35,
                0
            )

            # ------------------------------------------------
            # TITLE
            # ------------------------------------------------

            put_text(
                frame,
                "SEA WAVE COUNTER",
                (20, 35),
                0.9,
                2
            )

            # ------------------------------------------------
            # WAVE COUNT
            # ------------------------------------------------

            put_text(
                frame,
                f"WAVES: "
                f"{detector['wave_count']}",
                (20, 78),
                0.85,
                2
            )

            # ------------------------------------------------
            # SIGNAL
            # ------------------------------------------------

            put_text(
                frame,
                f"SIGNAL: "
                f"{smooth_signal:.4f}",
                (20, 108),
                0.55,
                1
            )

            # ------------------------------------------------
            # RIGHT STATUS
            # ------------------------------------------------

            put_text(
                frame,
                detector["status"],
                (width - 300, 45),
                0.6,
                2
            )

            put_text(
                frame,
                f"MOTION: "
                f"{current_motion:.3f}",
                (width - 300, 75),
                0.5,
                1
            )

            put_text(
                frame,
                f"FOAM: "
                f"{current_foam:.3f}",
                (width - 300, 102),
                0.5,
                1
            )

            # ------------------------------------------------
            # TIME
            # ------------------------------------------------

            mins = int(
                current_time // 60
            )

            secs = int(
                current_time % 60
            )

            total_mins = int(
                duration // 60
            )

            total_secs = int(
                duration % 60
            )

            put_text(
                frame,
                f"{mins:02d}:{secs:02d} / "
                f"{total_mins:02d}:{total_secs:02d}",
                (width - 175, height - 20),
                0.5,
                1
            )

            # ------------------------------------------------
            # CONTROLS
            # ------------------------------------------------

            put_text(
                frame,
                "SPACE Pause | R Restart | "
                "+ Faster | - Slower | Q Quit",
                (15, height - 20),
                0.5,
                1
            )

            # ------------------------------------------------
            # SHOW
            # ------------------------------------------------

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

        else:

            # ------------------------------------------------
            # PAUSED
            # ------------------------------------------------

            paused_frame = frame.copy()

            put_text(
                paused_frame,
                "PAUSED - PRESS SPACE",
                (
                    width // 2 - 170,
                    height // 2
                ),
                0.8,
                2
            )

            cv2.imshow(
                WINDOW_NAME,
                paused_frame
            )

        # ----------------------------------------------------
        # KEYBOARD
        # ----------------------------------------------------

        key = cv2.waitKey(delay) & 0xFF

        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        if (
            key == ord("q")
            or
            key == 27
        ):

            break

        # ----------------------------------------------------
        # PAUSE
        # ----------------------------------------------------

        elif key == 32:

            paused = not paused

            if paused:

                print("⏸ Paused")

            else:

                print("▶ Resumed")

        # ----------------------------------------------------
        # RESTART
        # ----------------------------------------------------

        elif key == ord("r"):

            print("🔄 Restarting...")

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            detector = reset_detector()

            frame_number = 0

        # ----------------------------------------------------
        # SPEED UP
        # ----------------------------------------------------

        elif (
            key == ord("+")
            or
            key == ord("=")
        ):

            delay = max(
                1,
                delay - 5
            )

        # ----------------------------------------------------
        # SPEED DOWN
        # ----------------------------------------------------

        elif key == ord("-"):

            delay += 5

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()

    # ========================================================
    # FINAL RESULT
    # ========================================================

    final_count = detector[
        "wave_count"
    ]

    print()
    print("=" * 70)
    print("🌊 SEA WAVE COUNTING COMPLETE")
    print("=" * 70)
    print()

    print(
        f"🌊 TOTAL WAVES COUNTED: "
        f"{final_count}"
    )

    print()

    print(
        "The counter analyzed the selected "
        "water region only."
    )

    print()

    print("=" * 70)

    # --------------------------------------------------------
    # FINAL RESULT WINDOW
    # --------------------------------------------------------

    result = np.zeros(
        (500, 900, 3),
        dtype=np.uint8
    )

    result[:] = (
        20,
        20,
        20
    )

    put_text(
        result,
        "SEA WAVE COUNTER",
        (235, 100),
        1.2,
        3
    )

    put_text(
        result,
        f"WAVES COUNTED: "
        f"{final_count}",
        (245, 200),
        1.0,
        2
    )

    put_text(
        result,
        "ANALYSIS COMPLETE",
        (285, 280),
        0.8,
        2
    )

    put_text(
        result,
        "Press any key to close.",
        (325, 400),
        0.55,
        1
    )

    cv2.imshow(
        "FINAL RESULT",
        result
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":

    main()