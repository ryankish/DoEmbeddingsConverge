import importlib
from box import Box

def load_experiment_config(experiment_name):
    try:
        module = importlib.import_module(f".{experiment_name}", package="configs")
        return getattr(module, "experiment_config")
    except ImportError:
        raise ValueError(f"Experiment configuration '{experiment_name}' not found.")
    except AttributeError:
        raise ValueError(f"Experiment configuration '{experiment_name}' is invalid.")
