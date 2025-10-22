import pandas as pd
import logging
import pickle
from catboost import CatBoostClassifier

# Настройка логгера
logger = logging.getLogger(__name__)

logger.info('Importing pretrained model...')

# Import model
model = CatBoostClassifier()
with open('./models/catboost.pickle', 'rb') as handle:
    model = pickle.load(handle)

# Define optimal threshold
model_th = 0.98
logger.info('Pretrained model imported successfully...')

# Make prediction
def make_pred(dt, path_to_file):

    # Make submission dataframe
    submission = pd.DataFrame({
        'index':  pd.read_csv(path_to_file).index,
        'prediction': (model.predict_proba(dt)[:, 1] > model_th) * 1
    })
    logger.info('Prediction complete for file: %s', path_to_file)

    # Return proba for positive class
    return submission, model.predict_proba(dt)[:, 1]
