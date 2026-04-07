import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    project_name: str = os.getenv("PROJECT_NAME", "hotel-review-project")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()