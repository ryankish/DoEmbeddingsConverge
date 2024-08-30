from box import Box

experiment1 = {
        'experiment_core': {
                'experiment_id': 1,
                'model1_embed_init': 'glorot_uniform',
                'model2_embed_init': 'glorot_uniform',
                'start_model_path': None,
                'lock_weights': False,
                'plot_title': 'Baseline Model Perplexity',
        },
        'data': {
            'dataset_seed': 0,
        },
        'model': {
            'd_model': 128,
            'n_layers': 6,
            'heads': 8,
            'norm': 2.0,
            'seqlen': 512,

        },
        'training1': {
            'batch_size': 3,
            'SGDR': False,
            'sched': None,
            'lr': 0.0001,
            'threshold': 3, # TODO: look at this
            'dropout': 0.1
        },
        'training2': {
            'epochs': 40,
            'verbose': False,
            'train_subset': None, # for testing purposes only
        },
        'device': {
            'no_cuda': False,
            'device': "cuda:0", 
        },
        'set_programmatically': {
            'train': None,
            'valid': None,
            'test': None,
            'optimizer': None,
        }
    }
config = Box(experiment1)
#
# experiment2 = {
#         'experiment_id': experiment_id,
#         'dataset_seed': 0,
#         'device': "cuda:0",
#         'no_cuda': False,
#         'SGDR': False,
#         'epochs': 2,
#         'model1_embed_init': model1_embed_init,
#         'model2_embed_init': model2_embed_init,
#         'd_model': experiment2_embedding_size,
#         'n_layers': 6,
#         'heads': 8,
#         'dropout': 0.1,
#         'batchsize': 3,
#         'printevery': 1, # TODO: implement
#         'lr': 0.00001,
#         'seqlen': 512,
#         'threshold': 3,
#         'norm': 2.0,
#         'verbose': False,
#         'time_name': None,
#         'train_subset': None, # for testing purposes only
#         'train': None,
#         'valid': None,
#         'test': None,
#         'optimizer': None,
#         'sched': None,
#         'plot_title': None,
#         'lock_weights': True,
#         'starter_model_path': 'experiments/1/models/1/checkpoint_40.pt',
#     }

