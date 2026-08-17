from redis import Redis
from rq import Queue, Worker

from api.config import REDIS_URL
from pipeline.results_repo import PostgresResultsRepository
from pipeline.runner import run_pipeline


def run_pipeline_job(filename: str) -> dict:
    return run_pipeline(filename)


def main() -> None:
    conn = Redis.from_url(REDIS_URL)
    PostgresResultsRepository().init_db()
    worker = Worker(Queue("svar", connection=conn))
    worker.work()


if __name__ == "__main__":
    main()