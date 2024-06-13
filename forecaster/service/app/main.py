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
from app.src.data_management import logger
import traceback
import requests
from datetime import datetime

app = FastAPI()

################################################################################################
########################################### ERROR MANAGE DEFINITION ############################
################################################################################################
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

################################################################################################
###########################################EOF definition#######################################
################################################################################################


def replace_null_with_none(data):
    """
    workaround for parsing a json to a configdict
    """
    if isinstance(data, dict):
        return {k: replace_null_with_none(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_null_with_none(item) for item in data]
    elif data == 'NULL':
        return None
    else:
        return data


def get_model_name(conf:DictConfig):
    """
    Define the name of the model combining unique values
    """
    return f'{conf.main.end_point.parameters.buildingName}_{conf.main.end_point.parameters.spaceName}_{conf.main.end_point.parameters.sourceId}_{conf.main.name}_{conf.main.version}'

def send_to_logbook(data:pd.DataFrame,conf:DictConfig):
    """Send prediction to the logbook

    Args:
        data (pd.DataFrame): data to send to the logbook
        conf (DictConfig): configurations containing the definition of the end point
    """
    
    content = []
    for index, row in data.iterrows():
        ##create the vector of the prediction: the non mandatory fields are in the metadata key
        content.append(dict(description= conf.main.end_point.parameters.content_description,
                            property = conf.main.end_point.parameters.property,
                            value = np.round(row['y'],2) ,
                            unitOfMeasure=conf.main.end_point.parameters.unitOfMeasure,
                            refDateTime = row['time'].strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
                            metadata = dict(y_low = np.round(row['y_low'],2),
                                        y_high = np.round(row['y_high'],2),
                                        lag = row['lag'],
                                        prediction_time =row['prediction_time'].strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
                            )
                            ))
    #compose the filan dictionary --> leave id empty the logbook will assign it during the uploading
    res = dict(id="",
                          type = conf.main.end_point.parameters.type,
                          description= conf.main.end_point.parameters.description,
                          dateTime=  datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
                          sourceId= conf.main.end_point.parameters.sourceId,
                          buildingName= conf.main.end_point.parameters.buildingName,
                          spaceName= conf.main.end_point.parameters.spaceName,
                          content=content
                          )


    headers = {'Content-Type': 'application/json'}
    res = json.dumps({'site': conf.main.end_point.parameters.site,'event':res})
    #for testing purposes you can avoid to send it to the logbook
    if conf.main.end_point.send_to_logbook:
        response = requests.request("POST", conf.main.end_point.result_url, headers=headers, data=res)
        logger.info(f'logbook sending response {response.json()}')

      



@app.get("/")
def read_root():
    return {"Hello": "World"}


def send_notification(conf:DictConfig,message:str):
    """Send notification to the logbook when the training procedure is completed

    Args:
        conf (DictConfig): configuration with end point specification
        message (str): message to send
    """
    to_send = {
            "id": "",
            "type": "Notification",
            "description": f"Model {get_model_name(conf)} trained correctly",
            "dateTime": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
            "sourceId": conf.main.end_point.parameters.sourceId,
            "buildingName": conf.main.end_point.parameters.buildingName,
            "spaceName": conf.main.end_point.parameters.spaceName,
            "content": [
            {
                "category": "Notification",
                "priority": "<Normal>",
                "topic": "Model trained notification",
                "text": message,
                "metadata": {
                    "destination": "Everyone"	
                }
            }
            ]
        }
        
    res = json.dumps({'site': conf.main.end_point.parameters.site,'event':to_send})
    headers = {'Content-Type': 'application/json'}
    response = requests.request("POST", conf.main.end_point.notification_url, headers=headers, data=res)
    logger.info(response)
    

##this is an auxiliary function for calling the train async
def train(model:Model, data:pd.DataFrame,conf:DictConfig):
    """function for training the model

    Args:
        model (Model): Model 
        data (pd.DataFrame): data used for the train/validation phase
        conf (DictConfig): configuration for the training phase
    """
    
    logger.info("##########################START TRAINING#############################")
    trained, loss = model.train_model(data,conf)
    if trained:
        message = f'Model successfully trained with a validation loss of {np.round(loss,4)}'
    else:
        message = 'Model not trained, see the logs'
        
        
    logger.info('##############END TRAINING############################################')
    send_notification(conf,message)


@app.post("/train/")
async def train_model(request: NotificationRequest,background_tasks: BackgroundTasks)->dict:
    """post API for training the model

    Args:
        request (NotificationRequest): request sent to the post
        background_tasks (BackgroundTasks): here we need the asyncronous part

    Raises:
        ConfigError: configuration is not correct
        ModelNotTrained: there are some errors while training the model
        DataNotLoaded: data can not be retrieve

    Returns:
        dict: {"message": "Train sent"} --> if no exception are raised it returns a standard message
    """
    try:
        params = replace_null_with_none(json.loads(request.params))
    except:
        raise ConfigError(request.params)
    #load omegaconf parameters
    conf = OmegaConf.create(params)
    model_name = get_model_name(conf) 
    dirpath = os.path.join(conf.main.main_folder,'weights',model_name)
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
    
    
    data = model.get_historical_data()
    if data is None:
        raise DataNotLoaded(conf)
    #Async job
    background_tasks.add_task(train, model,data, conf)
    return {"message": "Train sent"}

@app.get("/predict/")
def predict(request: NotificationRequest):
    """get API for getting the prediction

    Args:
        request (NotificationRequest): request with the parameters for the inference part. See the examples
        but keep in mind that end_time and start_time are important: if they are equal we obtain N predicted values
        where N is the number of steps define during the training procedure. In are different, we get a lot of prediction,
        because for a given timestamp I can have the predictions made in all the different steps before: in this case 
        we introduce the variable lag: if lag=1 it means is that the values is the first predicted value, if lag=20 it means
        that it is the 20-th value in the output vector.
        

    Raises:
        ConfigError: configuration is not correct
        ModelNotLoaded: model can not be loaded
        DataNotLoaded: data can not be retrieving
        ForecastNotPossible: there are some problems during the inference step

    Returns:
        json: resulting json with a unique key 'forecast' with all the variables
    """
    try:
        conf = replace_null_with_none(json.loads(request.params))
    except:
        raise ConfigError(request.params)
    #load inference parameters
    conf = OmegaConf.create(conf)
    
    ## load backbone model
    model_name = get_model_name(conf) 
    dirpath = os.path.join(conf.main.main_folder,'weights',model_name)

    try:
        model = load_model(dirpath,conf)
    except:
        raise ModelNotLoaded(conf)
    
    logger.info('Model loading ok!')
    ## load data
    #here we have to use only historical data, in a real application there is the call to the endpoint!
    ## TODO add endpoint retrival procedure
    
    try:
        data = model.get_historical_data()
    except:
        raise DataNotLoaded(conf)
    logger.info('Data retrieving ok!')

    ## load timeseries to the model and load the correct forecasting method
    model.prepare(data,conf)
    ## get the prediction
    try:
        res = model.inference(conf)
    except:
        raise ForecastNotPossible(conf)
    logger.info('Forecasting ok!')

  

    ##send the results to the logbook
    send_to_logbook(res,conf)

    # transform time for re the result
    res.time = res.time.dt.strftime('%Y-%m-%d %H:%M:%S')
    res.prediction_time = res.prediction_time.dt.strftime('%Y-%m-%d %H:%M:%S')
    return {"forecast": res[['time','lag','y','y_low','y_median','y_high','prediction_time']].to_json(orient='records')}

