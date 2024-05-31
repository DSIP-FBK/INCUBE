# Incube forecaster tool

This tool is meant to be used for training forecasting models inside the Incube project (grant agreement no. 101069610) and developed inside the tasks 5.1 and 7.2 by Andrea Gobbi, DSIP unit, DI center, Fondazione Bruno Kessler, Trento Italy (agobbi@fbk.eu).


## Description

Create a virtual environment for python, install pip and install the [DSIPTS library](https://github.com/DSIP-FBK/DSIPTS). You can install it from source (not suggested) or using pip:
```
pip install --force dsipts --index-url ${TOKEN_DSIPTS}
```
Ask to Andrea Gobbi for the token. There is the possibility to use a `csv` file as end point for testing the procedure. In this case ask to Andrea Gobbi the data and put the `csv` inside the `data` folder.


In this repository you can find:
1. a docker container for training AI models and get the predictions (`service` folder)
2. scripts for testing the models and the pipeline (`scripts` folder)
They should be equivalent, both uses the code inside the `service/src` folder, but scripts can be used from command line while the service is a docker container with FastAPI described in the [docker](#docker) section.


The training procedures follows three steps:
1. define the model using the class `Model`
2. retrieve historical data from the specified `end-point`
3. actually train the model (for now only [iTransformer](https://arxiv.org/abs/2310.06625)) using the [DSIPTS library](https://github.com/DSIP-FBK/DSIPTS).

The training procedure can be fired with the following commands:
```
cd scrips
python train.py  --config-dir=../config --config-name=config_toy
```

The inference procedure follows four steps:
1. load the model
2. get the inference data 
3. prepare the data
3. call the inference phase 

```
cd scrips
 python inference.py  --config-dir=../config --config-name=config_toy_inference 
```

The same functionalities are available using the docker container described in the following section

## Docker


After downloading the repo, go in the `service` folder and build the image. The process will try to accesses to the FBK gitlab, make sure to have the `.env` file (ask to agobbi@fbk.eu)
```
# get .env from FBK  
# get data from FBK (if csv end point will be used)
docker-compose build
#for debugging
docker-compose run --service-ports web 
#for real application
docker-compose up 
```
Open a new terminal, from the same virtual environment created in section [Description](#desciption) run python:

```
import requests
import json
from omegaconf import DictConfig, OmegaConf

#filename = '<path to config>/config_toy.yaml' 
filename = '/home/agobbi/Projects/Incube/forecaster/config/config_toy.yaml' 

url = "http://0.0.0.0:80/train/" #API for posting the training procedure. It is an asynchronous function but it will return the status ok if the training is performed. Once the training procedure ends it is possible to send a message to the logbook for updating the status (TODO).

p = str(OmegaConf.to_container(OmegaConf.load(filename))).replace("\'","\"").replace('True','true').replace('False','false').replace('None','\"NULL\"').replace('none','\"NULL\"') ##json workaround
params = {"params":  p}
response = requests.post(url, json=params)
print(response.status_code)
print(response.json())
```

Once the model is trained you can get the results:
```
import requests
from omegaconf import DictConfig, OmegaConf
url = "http://0.0.0.0:80/predict/"  #API for getting the results

#filename = '<path to config>/config_toy_inference.yaml' 
filename = '/home/agobbi/Projects/Incube/forecaster/config/config_toy_inference.yaml'

p = str(OmegaConf.to_container(OmegaConf.load(filename))).replace("\'","\"").replace('True','true').replace('False','false').replace('None','\"NULL\"').replace('none','\"NULL\"') ##json workaround
params = {"params":  p}
response = requests.get(url, json=params)
tmp = json.loads(response.json()['forecast']) ##the return of the get is a json with the unique key 'forecast'. TODO change according to the requirements of the logbook and Plenitude cloud service
res = pd.DataFrame(tmp)
```
You can plot also some statistics and an example of the results to be used inside the digital twin or by Plenitude.

```
import matplotlib.pyplot as plt
import numpy as np
ID=100
sample = res[res.prediction_time ==res.prediction_time.unique()[ID]]
plt.plot(sample.lag,sample.y,label ='real')
plt.plot(sample.lag,sample.y_median,'o-',label ='predicted')
plt.fill_between(sample.lag, sample.y_low, sample.y_high, color='lightblue', alpha=0.5,label='95% CI')
plt.title(f'Real vs prediction for prediction a time {res.prediction_time.unique()[ID]} ')
plt.xlabel('Lag')
plt.ylabel('Value')
plt.legend()
plt.show()

res['err'] = np.abs(res.y_median-res.y)
err = res.groupby('lag').err.mean().reset_index()
plt.plot(err.lag, err.err,'o-',label='MAE')
plt.title('MAE over lag times')
plt.xlabel('Lag')
plt.ylabel('MAE')
plt.show()
```




