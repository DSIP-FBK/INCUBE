import numpy as np
import sys
sys.path.append('../')
from service.app.src.data_management import Model
from omegaconf import DictConfig
import hydra
import os
import shutil
import logging

@hydra.main(version_base=None)
def train(conf: DictConfig) -> None:
    
    # define where to put weights loss and metadata for the models
    dirpath = os.path.join(conf.main.main_folder,'weights',conf.main.name, str(conf.main.version))
    if os.path.exists(dirpath):
        if conf.main.retrain:
            shutil.rmtree(dirpath)
    else:
        os.makedirs(dirpath)
        
    # define the model
    model = Model(name=conf.main.name,
                end_point=conf.main.end_point,
                main_folder=dirpath)

    # get historical data
    data = model.get_historical_data()

    # train the model
    trained, loss =model.train_model(data,conf)
    
    if trained:
        logging.info(f'Model trained with validation loss = {np.round(loss, 4)}')
    else:
        logging.info('Model not trained, see logs')

if __name__ == '__main__': 
    train()