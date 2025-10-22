import os
import sys
import pandas as pd
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime
import json

sys.path.append(os.path.abspath('./src'))
from scorer import make_pred, model
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProcessingService:
    def __init__(self):
        logger.info('Initializing ProcessingService...')
        self.input_dir = '/app/input'
        self.output_dir = '/app/output'
        # self.train = load_train_data()
        logger.info('Service initialized')

    def preprocess_df(self, data):
        data['transaction_time'] = pd.to_datetime(data['transaction_time'])
        data['hour'] = data['transaction_time'].dt.hour
        data['weekday'] = data['transaction_time'].dt.weekday
        data['distance'] = (data['lat'] - data['merchant_lat']) ** 2 + (data['lon'] - data['merchant_lon']) ** 2
        data = data.drop(columns=['transaction_time', 'name_1', 'name_2', 'lat', 'lon', 'merchant_lat', 'merchant_lon'])
        return data


    def process_single_file(self, file_path):
        try:
            logger.info('Processing file: %s', file_path)
            input_df = pd.read_csv(file_path)

            logger.info('Starting preprocessing')
            processed_df = self.preprocess_df(input_df)
            
            logger.info('Making prediction')
            submission, proba = make_pred(processed_df, file_path)
            
            logger.info('Prepraring submission file')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"predictions_{timestamp}_{os.path.basename(file_path)}"
            submission.to_csv(os.path.join(self.output_dir, output_filename), index=False)
            logger.info('Predictions saved to: %s', output_filename)

            output_filename = f"probability_distribution_{timestamp}_{os.path.basename(file_path)[:-4]}.png"
            plt.hist(proba)
            plt.title('Probability distribution')
            plt.savefig(os.path.join(self.output_dir, output_filename))
            logger.info('Probability distribution saved to: %s', output_filename)

            output_filename = f"feature_importance_{timestamp}_{os.path.basename(file_path)[:-4]}.json"
            importances = model.feature_importances_
            threshold = sorted(importances)[-5]
            res = {}
            for importance, name in zip(importances, model.feature_names_):
                if importance > threshold:
                    res[name] = importance

            with open(os.path.join(self.output_dir, output_filename), 'w') as outfile:
                json.dump(res, outfile)
            logger.info('Feature importance saved to: %s', output_filename)


        except Exception as e:
            logger.error('Error processing file %s: %s', file_path, e, exc_info=True)
            return


class FileHandler(FileSystemEventHandler):
    def __init__(self, service):
        self.service = service

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".csv"):
            logger.debug('New file detected: %s', event.src_path)
            self.service.process_single_file(event.src_path)

if __name__ == "__main__":
    logger.info('Starting ML scoring service...')
    service = ProcessingService()
    observer = Observer()
    observer.schedule(FileHandler(service), path=service.input_dir, recursive=False)
    observer.start()
    logger.info('File observer started')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info('Service stopped by user')
        observer.stop()
    observer.join()