
from dataclasses import dataclass
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from dsipts import TimeSeries, ITransformer ##here I expose only one model
import os
import numpy as np
import logging

@dataclass
class Model:
    """Class for managing predictive models in INCUBE"""
    name: str
    end_point: DictConfig
    main_folder:str
    data: pd.DataFrame = None
    trained: bool = False
    ts: TimeSeries = None
    train_config:DictConfig = None

    def get_historical_data(self)-> bool:
        """Get data from endpoint defined in the init or from a csv

        Returns:
            bool: True if the procedure ends correctly otherwise False
    
        """
        logging.info("###############LOADING DATA##################")
        if '.csv' in self.end_point.url:
            try:
                data = pd.read_csv(self.end_point.url)
            except:
                logging.info('Something went wrong while loading the csv')
                return False
        else:
            pass
            #here we need to retrieve data from the DBL or from PLENITUDE platform or Digital Twin
            ## maybe using parameters as additional argument
        if 'time' not in data.columns:
            logging.info('data must have the colum time')
            return False
        if 'y' not in data.columns:
            logging.info('data must have the colum y')
            return False
        data.time = pd.to_datetime(data.time)
        #self.data = data
        return data
    def train_model(self,data:pd.DataFrame,parameters:DictConfig)-> bool:
        """Train the predictive model

        Args:
            data (pd.DataFrame): data to use for training-validating the model
            parameters (DictConfig): parameters for the model, see the README 

        Returns:
            bool: True if the procedure ends correctly otherwise False
        """
        ts = TimeSeries(self.name)
        #ts.set_verbose(False)


        ts.load_signal(data,past_variables =parameters.timeseries.past_variables,
                       future_variables =parameters.timeseries.future_variables,
                       target_variables =['y'],
                       enrich_cat= parameters.timeseries.enrich_cat)
        
        ## create the model
        model_conf= parameters.model_configs
        model_conf['past_channels'] = len(ts.num_var)
        model_conf['future_channels'] = len(ts.future_variables)
        model_conf['embs'] = [ts.dataset[c].nunique() for c in ts.cat_var]
        model_conf['out_channels'] = len(ts.target_variables)
        logging.info(parameters.split_params)
        model =  ITransformer(**model_conf,   
                                optim_config = parameters.optim_config,
                                scheduler_config =parameters.scheduler_config,verbose=True ) 
         
        ## attach the model to the timeseries
        ts.set_model(model,config=dict(model_configs=model_conf,
                                       optim_config=parameters.optim_config,
                                       scheduler_config=parameters.scheduler_config
                                       ))
        split_params = parameters.split_params
        split_params['past_steps'] = model_conf.past_steps
        split_params['future_steps'] = model_conf.future_steps
        parameters.train_config.dirpath = self.main_folder
        ts.dirpath = self.main_folder    

        try:
            valid_loss = ts.train_model(split_params=split_params,**parameters.train_config)
            self.trained = True
        except:
            logging.info('can not train model')
            self.trained = False
            valid_loss = None
        if self.trained:
            ts.checkpoint_file_last = os.path.join(ts.dirpath ,'checkpoint.ckpt')
            ts.save(os.path.join(ts.dirpath ,'model'))
            
            with open(os.path.join(ts.dirpath,'config.yaml'),'w') as f:
                f.write(OmegaConf.to_yaml(parameters))
    
            
            
            
        return self.trained, valid_loss

    def inference(self,inference_parameters:DictConfig)-> bool:

        #self.ts = ts
        
        SD = np.datetime64(pd.to_datetime(inference_parameters.inference.start_date)-self.ts.freq*self.train_config.model_configs.past_steps).astype('datetime64[s]')
        ED =  np.datetime64(pd.to_datetime(inference_parameters.inference.end_date)+self.ts.freq*self.train_config.model_configs.future_steps).astype('datetime64[s]')
        
        self.ts.dataset.time = self.ts.dataset.time.apply(lambda x: x.tz_localize(None))
        res = self.ts.inference(data = self.ts.dataset[self.ts.dataset.time.between(SD,ED,inclusive='left')],steps_in_future=self.train_config.model_configs.future_steps)
        return res
    
    def prepare(self,data:str,inference_parameters:DictConfig)-> None:
        ts = TimeSeries(self.name)
        #ts.set_verbose(False)

        train_conf =  OmegaConf.load(os.path.join(self.main_folder,'config.yaml'))  
        ts.load_signal(data,past_variables =train_conf.timeseries.past_variables,
                       future_variables =train_conf.timeseries.future_variables,
                       target_variables =['y'],
                       enrich_cat= train_conf.timeseries.enrich_cat)
        
        ts.load(ITransformer,os.path.join(self.main_folder,'model'),load_last=inference_parameters.inference.load_last)
        self.ts = ts
        
        
def load_model(conf:DictConfig)->Model:
    dirpath = os.path.join(conf.main.main_folder,'weights',conf.main.name, str(conf.main.version))
    train_config =  OmegaConf.load(os.path.join(dirpath,'config.yaml'))     
    model = Model(name=train_config.main.name,
                end_point=train_config.main.end_point,
                
                main_folder=dirpath)
    model.train_config = train_config
    return model
    