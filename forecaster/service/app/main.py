from app.src.data_management import Model, load_model
from fastapi import FastAPI, BackgroundTasks,HTTPException,Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from omegaconf import  OmegaConf,DictConfig
from pydantic import BaseModel
import json
import pandas as pd
import numpy as np
import os
import shutil
import logging
import traceback


logger = logging.getLogger("uvicorn.error")
app = FastAPI()


########################################### ROUTINE DEFINITION ############################

def replace_null_with_none(data):
    if isinstance(data, dict):
        return {k: replace_null_with_none(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_null_with_none(item) for item in data]
    elif data == 'NULL':
        return None
    else:
        return data


class NotificationRequest(BaseModel):
    params: str




async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )

class ConfigError(Exception):
    def __init__(self, config: str):
        self.config = config

@app.exception_handler(ConfigError)
async def config_error_handler(request: Request, exc: ConfigError):
    logger.error(f"Configuration {exc.name} not correct")
    return JSONResponse(
        status_code=404,
        content={"message": "Configuration error", "content":traceback.format_exc()},
    )



class ModelNotTrained(Exception):
    def __init__(self, name: str):
        self.name = name

@app.exception_handler(ModelNotTrained)
async def model_not_trained_handler(request: Request, exc: ModelNotTrained):
    logger.error(f"Model {exc.name} not trained")
    return JSONResponse(
        status_code=404,
        content={"message": "Model not successfully trained", "content":traceback.format_exc()},
    )

class ModelNotLoaded(Exception):
    def __init__(self, name: str):
        self.name = name

@app.exception_handler(ModelNotLoaded)
async def model_not_found_handler(request: Request, exc: ModelNotLoaded):
    logger.error(f"Model {exc.name} not loaded")
    return JSONResponse(
        status_code=404,
        content={"message": "Model not successfully loaded", "content":traceback.format_exc()},
    )

class DataNotLoaded(Exception):
    def __init__(self, name: str):
        self.name = name

@app.exception_handler(DataNotLoaded)
async def data_not_found_handler(request: Request, exc: DataNotLoaded):
    logger.error(f"Model {exc.name} not loaded")
    return JSONResponse(
        status_code=404,
        content={"message": "DataNotLoaded", "content":traceback.format_exc()},
    )

class ForecastNotPossible(Exception):
    def __init__(self, name: str):
        self.name = name

@app.exception_handler(ForecastNotPossible)
async def forecast_not_possible_handler(request: Request, exc: ForecastNotPossible):
    logger.error(f"Model {exc.name} can not make prediction, check input data")
    return JSONResponse(
        status_code=404,
        content={"message": "Forecasts not available", "content":traceback.format_exc()},
    )


###########################################EOF definition###################################################

@app.get("/")
def read_root():
    return {"Hello": "World"}


def send_notification(conf:DictConfig,message:str):
    #TODO send somewhere the notification that the model has been trained
    logger.info(message)
    pass

##this is an auxiliary function for calling the train async
def train(model:Model, data:pd.DataFrame,conf:DictConfig):
    
    logger.info("##########################START TRAINING#############################")

    trained, loss = model.train_model(data,conf)
    if trained:
        message = f'Model successfully trained with a validation loss of {np.round(loss,4)}'
    else:
        message = 'Model not trained, see the logs'
        
        
    logger.info('##############END TRAINING############################################')
    send_notification(conf,message)


@app.post("/train/")
async def train_model(request: NotificationRequest,background_tasks: BackgroundTasks):
    try:
        params = replace_null_with_none(json.loads(request.params))
    except:
        raise ConfigError(request.params)
    #load omegaconf parameters
    conf = OmegaConf.create(params)
    dirpath = os.path.join(conf.main.main_folder,'weights',conf.main.name, str(conf.main.version))
    # if we allow the retraining it will drop the existing folder

    if os.path.exists(dirpath):
        if conf.main.retrain:
            shutil.rmtree(dirpath)
    else:
        os.makedirs(dirpath)
    
    ##setup the Model class
    try:
        model = Model(name=conf.main.name,
                    end_point=conf.main.end_point,
                    main_folder=dirpath)
    except:
        raise ModelNotTrained(dirpath)
    
    try:
        data = model.get_historical_data()
    except:
        raise DataNotLoaded(conf)
    #Async job
    background_tasks.add_task(train, model,data, conf)
    return {"message": "Train sent"}

@app.get("/predict/")
def predict(request: NotificationRequest):
    try:
        params = replace_null_with_none(json.loads(request.params))
    except:
        raise ConfigError(request.params)
    #load inference parameters
    inference_parameters = OmegaConf.create(params)
    
    ## load backbone model
    try:
        model = load_model(inference_parameters)
    except:
        raise ModelNotLoaded(params)
    
    logger.info('Model loading ok!')
    ## load data
    #here we have to use only historical data, in a real application there is the call to the endpoint!
    ## TODO add endpoint retrival procedure
    
    try:
        data = model.get_historical_data()
    except:
        raise DataNotLoaded(params)
    logger.info('Data retrieving ok!')

    ## load timeseries to the model and load the correct forecasting method
    model.prepare(data,inference_parameters)
    ## get the prediction
    try:
        res = model.inference(inference_parameters)
    except:
        raise ForecastNotPossible(params)
    logger.info('Forecasting ok!')

    res.time = res.time.dt.strftime('%Y-%m-%d %H:%M:%S')
    res.prediction_time = res.prediction_time.dt.strftime('%Y-%m-%d %H:%M:%S')

    return {"forecast": res[['time','lag','y','y_low','y_median','y_high','prediction_time']].to_json(orient='records')}

