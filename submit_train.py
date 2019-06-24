import submitit
from train_classifier import main
import os

datasets = [("ImageNet", 1000), ]  # ("CIFAR10", 10), ("CIFAR100", 100)]
noises = ["Normal"]  # [None,"Laplace","Normal"]
# ################################################################### Baseline +RT models #####################################################################################

# # Baseline+RT
for d in datasets:
    for n in noises:
        path_models = "models"
        path_logs = "logs"
        path_models = os.path.join(path_models, d[0])
        path_logs = os.path.join(path_logs, d[0])
        if n is None:
            path_models = os.path.join(path_models, "baseline")
            path_logs = os.path.join(path_logs, "baseline")

        else:
            path_models = os.path.join(path_models, "models_RT", n)
            path_logs = os.path.join(path_logs, "models_RT", n)
        executor = submitit.AutoExecutor(folder=path_logs)  # submission interface (logs are dumped in the folder)
        # "uninterrupted")  # timeout in min
        executor.update_parameters(gpus_per_node=2, timeout_min=4320, partition="dev")
        job = executor.submit(main, path_model=path_models,
                              dataset=d[0], num_classes=d[1],
                              epochs=200, batch_size=256,
                              resume_epoch=0, save_frequency=2,
                              adversarial_training=None, attack_list=[],
                              eot_samples=1,
                              noise=n, sigma=0.25)  # will compute add(5, 7)
        print(job.job_id)  # ID of your job


# ################################################################### AT models #####################################################################################
datasets = [("CIFAR10", 10), ("CIFAR100", 100)]
attacks = ["PGDL2"]  # ["CW", "PGDLinf", "PGDL2"]

for d in datasets:
    for a in attacks:
        path_models = "models"
        path_logs = "logs"
        path_models = os.path.join(path_models, d[0])
        path_logs = os.path.join(path_logs, d[0])
        path_models = os.path.join(path_models, "models_AT", a)
        path_logs = os.path.join(path_logs, "models_AT", a)
        executor = submitit.AutoExecutor(folder=path_logs)  # submission interface (logs are dumped in the folder)
        # "uninterrupted")  # timeout in min
        executor.update_parameters(gpus_per_node=8, timeout_min=4320, partition="uninterrupted")
        job = executor.submit(main, path_model=path_models,
                              dataset=d[0], num_classes=d[1],
                              epochs=200, batch_size=256,
                              resume_epoch=0, save_frequency=2,
                              adversarial_training="Single", attack_list=[a],
                              eot_samples=1,
                              noise=None, sigma=0.25)  # will compute add(5, 7)
        print(job.job_id)  # ID of your job


################################################################### RAT models #####################################################################################

datasets = [("CIFAR10", 10), ("CIFAR100", 100)]
attacks = ["PGDL2"]  # ["PGDLinf", "PGDL2"]
noises = ["Laplace", "Normal"]

for d in datasets:
    for a in attacks:
        for n in noises:
            path_models = "models"
            path_logs = "logs"
            path_models = os.path.join(path_models, d[0])
            path_logs = os.path.join(path_logs, d[0])
            path_models = os.path.join(path_models, "models_RAT", n+'_'+a)
            path_logs = os.path.join(path_logs, "models_RAT", n+'_'+a)
            # submission interface (logs are dumped in the folder)
            executor = submitit.AutoExecutor(folder=path_logs)
            # "uninterrupted")  # timeout in min
            executor.update_parameters(gpus_per_node=8, timeout_min=4320, partition="uninterrupted")
            job = executor.submit(main, path_model=path_models,
                                  dataset=d[0], num_classes=d[1],
                                  epochs=200, batch_size=256,
                                  resume_epoch=0, save_frequency=2,
                                  adversarial_training="Single", attack_list=[a],
                                  eot_samples=1,
                                  noise=n, sigma=0.25)  # will compute add(5, 7)
            print(job.job_id)  # ID of your job

################################################################### MAT models #####################################################################################


datasets = [("CIFAR10", 10), ("CIFAR100", 100)]
mix_strategy = ["MixMean", "MixMax", "MixRand"]

for d in datasets:
    for m in mix_strategy:
        path_models = "models"
        path_logs = "logs"
        path_models = os.path.join(path_models, d[0])
        path_logs = os.path.join(path_logs, d[0])
        path_models = os.path.join(path_models, "models_MAT", m)
        path_logs = os.path.join(path_logs, "models_MAT", m)
        # submission interface (logs are dumped in the folder)
        executor = submitit.AutoExecutor(folder=path_logs)
        # "uninterrupted")  # timeout in min
        executor.update_parameters(gpus_per_node=8, timeout_min=4320, partition="uninterrupted")
        job = executor.submit(main, path_model=path_models,
                              dataset=d[0], num_classes=d[1],
                              epochs=200, batch_size=256,
                              resume_epoch=0, save_frequency=2,
                              adversarial_training=m, attack_list=["PGDL1", "PGDLinf", "PGDL2"],
                              eot_samples=1,
                              noise=None, sigma=0.25)  # will compute add(5, 7)
        print(job.job_id)  # ID of your job


################################################################### RMAT models #####################################################################################


datasets = [("CIFAR10", 10), ("CIFAR100", 100)]
mix_strategy = ["MixMean", "MixMax", "MixRand"]
noises = ["Laplace", "Normal"]

for d in datasets:
    for m in mix_strategy:
        for n in noises:
            path_models = "models"
            path_logs = "logs"
            path_models = os.path.join(path_models, d[0])
            path_logs = os.path.join(path_logs, d[0])
            path_models = os.path.join(path_models, "models_RMAT", n+"_"+m)
            path_logs = os.path.join(path_logs, "models_RMAT", n+"_"+m)
            # submission interface (logs are dumped in the folder)
            executor = submitit.AutoExecutor(folder=path_logs)
            # "uninterrupted")  # timeout in min
            executor.update_parameters(gpus_per_node=8, timeout_min=4320, partition="uninterrupted")
            job = executor.submit(main, path_model=path_models,
                                  dataset=d[0], num_classes=d[1],
                                  epochs=200, batch_size=256,
                                  resume_epoch=0, save_frequency=2,
                                  adversarial_training=m, attack_list=["PGDL1", "PGDLinf", "PGDL2"],
                                  eot_samples=1,
                                  noise=None, sigma=0.25)  # will compute add(5, 7)
            print(job.job_id)  # ID of your job
