from pydantic_settings import BaseSettings,SettingsConfigDict

class Settings(BaseSettings):
    INDEX_NAME: str
    EMBEDDINGS_PROVIDER: str
    MODEL_PROVIDER: str
    VECTOR_DB_PROVIDER: str 
    EMBEDDING_API_KEY: str
    EMBEDDING_MODEL: str
    VECTOR_DB_API_KEY: str
    MODEL_API_KEY: str
    GENERATION_MODEL:str
    GRADER_MODEL:str
    GRADER_API_KEY:str
    GRADER_BASE_URL: str
    DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE: str
    DEEPEVAL_RETRY_MAX_ATTEMPTS: str
    DEEPEVAL_RETRY_CAP_SECONDS: str
    DEEPEVAL_RETRY_INITIAL_SECONDS:str
    model_config = SettingsConfigDict(
        env_file=".env"
    )


settings = Settings()
