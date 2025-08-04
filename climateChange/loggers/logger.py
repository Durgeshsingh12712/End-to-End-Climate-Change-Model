import os, sys, logging
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="%(asctime)s | %(name)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s",
    level=logging.INFO
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s")
console_handler.setFormatter(formatter)

logger = logging.getLogger("Climate Change Logger")
logger.addHandler(console_handler)