# Copyright 2023 [PT BOOKBOT INDONESIA](https://bookbot.id/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from datetime import datetime
import sys

class Logger:
    _instance = None
    _logger = None

    @classmethod
    def setup(cls, script_name: str = None, log_dir: str = "logs") -> logging.Logger:
        """Setup logger with file and console handlers"""
        # Only setup once
        if cls._logger is not None:
            return cls._logger

        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Generate log filename based on current datetime and script name
        timestamp = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        if script_name:
            script_name = os.path.basename(script_name).replace('.sh', '')
        log_filename = f"{timestamp}-speechline.log"
        log_path = os.path.join(log_dir, log_filename)

        # Create logger
        cls._logger = logging.getLogger(script_name if script_name else 'speechline')
        cls._logger.setLevel(logging.INFO)

        # Create handlers
        file_handler = logging.FileHandler(log_path)
        console_handler = logging.StreamHandler(sys.stdout)

        # Create formatters and add it to handlers
        log_format = logging.Formatter(f'%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)

        # Add handlers to the logger
        cls._logger.addHandler(file_handler)
        cls._logger.addHandler(console_handler)

        return cls._logger

    @classmethod
    def get_logger(cls) -> logging.Logger:
        """Get the logger instance, creating it if necessary"""
        if cls._logger is None:
            cls._logger = cls.setup()
        return cls._logger

# Create default logger instance
# logger = Logger.get_logger()
