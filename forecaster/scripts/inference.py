import sys
sys.path.append('../')
from service.app.src.data_management import load_model
from omegaconf import DictConfig
import logging
import hydra
import os

@hydra.main(version_base=None)
def inference(conf: DictConfig) -> None:
    dirpath = os.path.join(conf.main.main_folder,'weights',conf.main.name, str(conf.main.version))
    model = load_model(dirpath,conf)
    data = model.get_historical_data()
    model.prepare(data,conf)
    
    res = model.inference(conf)
    logging.info(res.head())


if __name__ == '__main__': 
    inference()