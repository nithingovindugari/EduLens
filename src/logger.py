import logging
import os
from datetime import datetime


LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs" , LOG_FILE)
os.makedirs(logs_path, exist_ok=True)


LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

print(LOG_FILE)
print(logs_path)
print(LOG_FILE_PATH)

logging.basicConfig(filename=LOG_FILE_PATH, 
                    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s", level=logging.INFO, )



if __name__ == "__main__":
    logging.info(" logging is started")

"""import logging:

This line imports the built-in logging module, which provides flexible and powerful logging capabilities in Python.
import os:

This line imports the built-in os module, which allows you to interact with the operating system, including creating directories and file paths.
from datetime import datetime:

This line imports the datetime class from the datetime module. It's used to work with dates and times.
log_file = f"datetime.now().strftime('%M_%d_%Y_%H_%M_%S')_log":

This line creates a string log_file that contains the current date and time formatted as minutes, day, year, hour, minute, and second. This string will be part of the log file's name.
logs_path = os.path.join(os.getcwd(), "logs", log_file):

This line constructs a path to a directory where the log files will be stored. It uses os.getcwd() to get the current working directory and appends "logs" and the log_file to create the complete path.
os.makedirs(logs_path, exist_ok=True):

This line creates the directory specified by logs_path. The exist_ok=True argument ensures that the directory is created if it doesn't exist, or it does nothing if the directory already exists.
log_file_path = os.path.join(logs_path, log_file):

This line constructs the complete file path for the log file by joining logs_path and log_file.
logging.basicConfig(filename=log_file_path, level=logging.INFO, format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"):

This line configures the logging system:
filename=log_file_path: Sets the log file where log records will be written. It uses the log_file_path constructed earlier.
level=logging.INFO: Sets the logging level to INFO, which means it will log messages at INFO level and above.
format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s": Defines the format of each log record. It includes the timestamp (%(asctime)s), line number (%(lineno)d), logger name (%(name)s), log level (%(levelname)s), and the log message itself (%(message)s)."""