from box import Box

# save_dir = ''
save_dir = '/data/ryan/other'
device = "cuda:5"

experiment_0 = Box(
    {
        "core": {
            "experiment_id": 0,
            "base_init_seed": 0,
            "model1_embed_init_seed": 1,
            # "model2_embed_init_seed": 2,
            # "model3_embed_inti_seed": 1,
            "model1_embed_init": "glorot_uniform",
            # "model2_embed_init": "glorot_uniform",
            # "model3_embed_init": "glorot_uniform",
            "start_modeler_path": None,
            "lock_weights": False,
            "plot_title": "Baseline Model Perplexity",
        },
        "run": {
            "test_dataloader": False,
        },
        "model": {
            "vocab_size": 50257,
            "d_model": 512,
            "n_layers": 6,
            "heads": 8,
            "norm": 2.0,
            "seqlen": 512,
        },
        "training1": {
            "batch_size": 12,
            "SGDR": False,
            "sched": None,
            "lr": 0.0001,
            "dropout": 0.1,
        },
        "training2": {
            "epochs": 40,
            "verbose": False,
            "dev_subset": None,  # for testing purposes only
        },
        "device": "cuda:1",
        "save_dir": save_dir
    }
)

experiment_1 = Box(
    {
        "core": {
            "experiment_id": 1,
            "base_init_seed": experiment_0.core.base_init_seed,
            "model1_embed_init_seed": 1,
            "model2_embed_init_seed": 2,
            # "model3_embed_inti_seed": 3,
            "model1_embed_init": experiment_0.core.model1_embed_init,
            "model2_embed_init": experiment_0.core.model1_embed_init,
            # "model3_embed_init": 1,
            "starter_model_path": None,
            "lock_weights": False,
            "plot_title": "Experiment 1 Perplexity",
        },
        "run": {
            "test_dataloader": False,
        },
        "model": {
            "vocab_size": experiment_0.model.vocab_size,
            "d_model": experiment_0.model.d_model,
            "n_layers": experiment_0.model.n_layers,
            "heads": experiment_0.model.heads,
            "norm": experiment_0.model.norm,
            "seqlen": experiment_0.model.seqlen,
        },
        "training1": {
            "batch_size": experiment_0.training1.batch_size,
            "SGDR": experiment_0.training1.SGDR,
            "sched": experiment_0.training1.sched,
            "lr": experiment_0.training1.lr,
            "dropout": experiment_0.training1.dropout,
        },
        "training2": {
            "epochs": 40,
            "dev_subset": None,
        },
        "device": device,
        "save_dir": save_dir
    }
)

experiment_2 = Box(
    {
        "core": {
            "experiment_id": 2,
            "base_init_seed": experiment_0.core.base_init_seed,
            "model1_embed_init_seed": 100,
            "model2_embed_init_seed": 200,
            # "model3_embed_inti_seed": 3,
            "model1_embed_init": experiment_0.core.model1_embed_init,
            "model2_embed_init": experiment_0.core.model1_embed_init,
            # "model3_embed_init": 1,
            "starter_model_path": "experiments/0/models/1/ckpts/ckpt_7.pt",
            "lock_weights": True,
            "plot_title": "Experiment 2 Perplexity",
        },
        "run": {
            "test_dataloader": False,
        },
        "model": {
            "vocab_size": experiment_0.model.vocab_size,
            "d_model": experiment_0.model.d_model,
            "n_layers": experiment_0.model.n_layers,
            "heads": experiment_0.model.heads,
            "norm": experiment_0.model.norm,
            "seqlen": experiment_0.model.seqlen,
        },
        "training1": {
            "batch_size": experiment_0.training1.batch_size,
            "SGDR": experiment_0.training1.SGDR,
            "sched": experiment_0.training1.sched,
            "lr": experiment_0.training1.lr,
            "dropout": experiment_0.training1.dropout,
        },
        "training2": {
            "epochs": 2,
            "dev_subset": 12,
        },
        "device": device,
        "save_dir": save_dir
    }
)
