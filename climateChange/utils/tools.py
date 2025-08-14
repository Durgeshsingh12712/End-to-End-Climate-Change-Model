import os, sys, joblib, yaml, json, joblib
from pathlib import Path
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations

from sklearn.metrics import(
    accuracy_score,
    mean_absolute_error, 
    r2_score
)
from sklearn.model_selection import GridSearchCV

from climateChange.loggers import logger
from climateChange.exceptions import CCException

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads yaml file and returns ConfigBox object
    
    Args:
        path_to_yaml (Path): path like input
        
    Raises:
        ValueError: if yaml file is empty
        e: empty file
        
    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"Yaml File: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("Yaml File is Empty")
    except Exception as e:
        logger.error(f"Error reading yaml file: {e}")
        raise e
    
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok= True)
        if verbose:
            logger.info(f"Create Directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Save json data
    
    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    
    logger.info(f"Json File saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Load json files data
    
    Args:
        path (Path): path to json file
        
    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path, 'r') as f:
        contnet = json.load(f)

    logger.info(f"JSON File loaded successfully from: {path}")
    return ConfigBox(contnet)

@ensure_annotations
def save_object(data, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")

@ensure_annotations
def load_object(path: Path):
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """

    data = joblib.load(path)
    logger.info(f"Binary file loaded from : {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """
    Get size in KB
    
    Args:
        path (Path): path of the file
        
    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"
    
def evaluate_models(X_train, y_train, X_test, y_test, models: dict, param: dict):
    """
    Evaluate multiple models and return results
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features  
        y_test: Test target
        models: Dictionary of models to evaluate
        param: Dictionary of parameters for each model
        
    Returns:
        dict: Model evaluation results
    """
    try:
        report = {}
        
        for model_name, model in models.items():
            para = param[model_name]
            
            gs = GridSearchCV(model, para, cv=3, scoring='neg_mean_absolute_error')
            gs.fit(X_train, y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Use regression metrics only
            report[model_name] = {
                'train_r2': r2_score(y_train, y_train_pred),
                'test_r2': r2_score(y_test, y_test_pred),
                'train_mae': mean_absolute_error(y_train, y_train_pred),
                'test_mae': mean_absolute_error(y_test, y_test_pred),
                'best_params': gs.best_params_
            }
            
        return report
    
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        raise CCException(e, sys)
