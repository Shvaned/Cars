import time
import os
from datetime import datetime
from PIL import ImageGrab

SAVE_FOLDER = "saved_images"
CHECK_INTERVAL = 1  # seconds

def ensure_save_folder():
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

def save_image_from_clipboard():
    image = ImageGrab.grabclipboard()
    if image:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_FOLDER, f"clipboard_{timestamp}.png")
        image.save(filename, "PNG")
        print(f"Saved image: {filename}")
        return True
    return False

def monitor_clipboard():
    print("Monitoring clipboard for images... Press Ctrl+C to stop.")
    last_saved = None
    while True:
        try:
            image = ImageGrab.grabclipboard()
            if image and image != last_saved:
                if save_image_from_clipboard():
                    last_saved = image
            time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            print("\nStopped.")
            break

if __name__ == "__main__":
    ensure_save_folder()
    monitor_clipboard()
