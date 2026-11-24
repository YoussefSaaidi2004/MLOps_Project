"""
deploiement_prefect.py
Déploie les flows avec une planification quotidienne (Prefect 3, serve()).

Terminal 1 : prefect server start
Terminal 2 : prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
             python deploiement_prefect.py
Terminal 3 : prefect deployment ls
             prefect deployment run 'all/ml-pipeline-all'
"""

from prefect import serve

from pipeline_prefect import all_steps, code, evaluate, install, train

if __name__ == "__main__":
    deployments = [
        all_steps.to_deployment(
            name="ml-pipeline-all",
            cron="0 2 * * *",  # tous les jours à 02:00
            tags=["full-pipeline", "mlops"],
        ),
        train.to_deployment(name="ml-pipeline-train", tags=["training", "mlops"]),
        evaluate.to_deployment(name="ml-pipeline-evaluate", tags=["evaluation", "mlops"]),
        code.to_deployment(name="ml-pipeline-code", tags=["code", "mlops"]),
        install.to_deployment(name="ml-pipeline-install", tags=["setup", "mlops"]),
    ]
    serve(*deployments)
