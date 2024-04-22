from datetime import timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime
from tweets_sentiment_analysis import run_tweet_etl

current_date = datetime.now()
start_date_formatted = current_date.strftime('%Y, %m, %d')    

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': start_date_formatted,
    'email': ['lokeshdammalapati@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

dag = DAG(
    'twitter_dag',
    default_args=default_args,
    description='Our first DAG with ETL process!',
    schedule_interval=timedelta(days=1),
)

run_etl = PythonOperator(
    task_id='complete_twitter_etl',
    python_callable=run_tweet_etl,
    dag=dag, 
)

run_etl
