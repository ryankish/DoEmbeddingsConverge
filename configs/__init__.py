from . import configs


def load_experiment_config(experiment_num):
    experiment_name = f"experiment_{experiment_num}"
    try:
        return getattr(configs, experiment_name)
    except AttributeError:
        raise ValueError(f"Experiment configuration '{experiment_name}' not found.")
