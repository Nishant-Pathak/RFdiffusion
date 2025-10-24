import os


def is_production_environment() -> bool:
	return os.getenv("ENVIRONMENT", "development") == "production"
