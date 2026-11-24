import logging
import yaml
import argparse
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
from pathlib import Path
from crackseg.datamodule import CrackDataModule
from crackseg.model import CrackSegModel
from crackseg.logging_utils import setup_logging
from datetime import datetime


logger = logging.getLogger(__name__)   #basic logger for logging trackebacks

def load_yaml_config(yaml_path: Path) -> dict:
    '''safely parse yaml based parameters'''

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_encoded = yaml.safe_load(f)
            return yaml_encoded

    except yaml.YAMLError as exc:
        logger.exception("Failed to parse config %s", yaml_path)


def build_mlf_logger(ml_config: dict) -> MLFlowLogger:
    '''utility function to build ml logger based on dict'''
    mlf = MLFlowLogger(**ml_config)
    return mlf

def build_parser() -> argparse.ArgumentParser:
    '''utility function to build parser based on args'''
    parser = argparse.ArgumentParser(description="CLI for model training")
    parser.add_argument("config", nargs="?", default="configs/base.yaml")
    return parser

def build_callbacks(cfg: dict) -> list[L.Callback]:
    '''utility function to build callbacks based on args'''
    cfg = cfg["callbacks"]
    cfg_monitor = cfg["monitor"]
    cfg_mode = cfg["mode"]
    cfg_save = cfg["save_top_k"]
    cfg_patience = cfg["patience"]

    checkpoint = ModelCheckpoint(monitor=cfg_monitor, mode=cfg_mode, save_top_k=cfg_save)
    early_stopping = EarlyStopping(monitor=cfg_monitor, mode=cfg_mode, patience=cfg_patience)

    return [checkpoint, early_stopping]

def main():
    '''main function for running model training/inference'''

    log_file = Path("logs")/f"train{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    setup_logging(level="INFO", log_file=log_file)   #setup general logging service
    logger.info("Logging to %s", log_file)
    parser = build_parser()


    args = parser.parse_args() #arguments from CLI
    cfg = load_yaml_config(args.config)
    mlf = build_mlf_logger(cfg["mlflow"])  # setup Mlflow logger for experiments logging
    mlf.experiment.log_artifact(mlf.run_id, args.config)

    L.seed_everything(cfg["seed"]) #ustawienie stałego seeda dla powtarzalnych ekspermetów

    callbacks = build_callbacks(cfg)
    dm = CrackDataModule(**cfg["data"])  #wczytanie DataModule
    model = CrackSegModel(**cfg["model"]) #Wczytanie Archiketury/modelu
    trainer = L.Trainer(**cfg["trainer"], logger = mlf, callbacks = callbacks)

    trainer.fit(model, dm) #treining
    trainer.test(model, dm,  ckpt_path="best") #test



if __name__ == "__main__":
    main()