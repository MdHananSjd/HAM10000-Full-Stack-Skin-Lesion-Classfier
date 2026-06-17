import logging
import os
from datetime import datetime

LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_training.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# output presentation
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()  # printing to the terminal terminal
    ]
)

# instantiate global tracking module
logger = logging.getLogger("SkinLesionClassifier")