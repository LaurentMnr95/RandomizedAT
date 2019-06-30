import submitit
from test_classifier import main
import os
from utils import *

attacks = [None, "PGDLinf", "PGDL2", "PGDL1"]
#################################################################### Baseline models #####################################################################################
# datasets = [("CIFAR10", 10), ("CIFAR100", 100)]
# # TODO add rm result file
# # no noise at inference baseline
# for d in datasets:
#     for a in attacks:
#         path_models = "models_copy/models"
#         path_logs = "logs_attacks"
#         path_results = "results_attacks"

#         path_models = os.path.join(path_models, d[0])
#         path_logs = os.path.join(path_logs, d[0])
#         path_results = os.path.join(path_results, d[0])

#         path_models = os.path.join(path_models, "baseline")
#         path_logs = os.path.join(path_logs, "baseline")
#         path_results = os.path.join(path_results, "baseline")

#         if not os.path.exists(path_results):
#             os.makedirs(path_results)

#         result_file = "attacks"
#         path_results = os.path.join(path_results, result_file)
#         if os.path.exists(path_results+".txt"):
#             delete_line(path_results+".txt", a)

#         # submission interface (logs are dumped in the folder)
#         executor = submitit.AutoExecutor(folder=path_logs)  # "uninterrupted")  # timeout in min
#         gpus_per_node = 4
#         executor.update_parameters(nodes=1, gpus_per_node=gpus_per_node, tasks_per_node=gpus_per_node,
#                                    cpus_per_task=int(80/gpus_per_node), timeout_min=4320,
#                                    partition="dev", comment="Deadline ACML 15/07", mem_gb=256, constraint="volta32gb")
#         init_file = get_init_file().as_uri()

#         job = executor.submit(main, init_file=init_file, path_model=os.path.join(path_models, "BEST.t7"),
#                               result_file=path_results,
#                               dataset=d[0], num_classes=d[1],
#                               batch_size=512,
#                               attack=a, eot_samples=1,
#                               noise=None,
#                               batch_prediction=1, sigma=0.25)
#         print(job.job_id)

# #################################################################### RT models #####################################################################################
# datasets = [("CIFAR10", 10), ("CIFAR100", 100)]
# noises = ["Laplace", "Normal"]
# eot_samples = [80]

# for a in attacks:
#     for d in datasets:
#         for n in noises:
#             for e in eot_samples:

#                 path_models = "models_copy/models"
#                 path_logs = "logs_attacks"
#                 path_results = "results_attacks"

#                 path_models = os.path.join(path_models, d[0])
#                 path_logs = os.path.join(path_logs, d[0])
#                 path_results = os.path.join(path_results, d[0])

#                 path_models = os.path.join(path_models, "models_RT", n)
#                 path_logs = os.path.join(path_logs, "models_RT", n)
#                 path_results = os.path.join(path_results, "models_RT", n)

#                 if not os.path.exists(path_results):
#                     os.makedirs(path_results)

#                 result_file = "attacks"
#                 path_results = os.path.join(path_results, result_file)

#                 if os.path.exists(path_results+".txt"):
#                     delete_line(path_results+".txt", [a, str(e)])

#                 # submission interface (logs are dumped in the folder)
#                 executor = submitit.AutoExecutor(folder=path_logs)
#                 # "uninterrupted")  # timeout in min
#                 gpus_per_node = 8
#                 executor.update_parameters(nodes=1, gpus_per_node=gpus_per_node, tasks_per_node=gpus_per_node,
#                                            cpus_per_task=int(80/gpus_per_node), timeout_min=4320,
#                                            partition="priority", comment="Deadline ACML 15/07", mem_gb=256, constraint="volta32gb")
#                 init_file = get_init_file().as_uri()

#                 job = executor.submit(main, init_file=init_file, path_model=os.path.join(path_models, "BEST.t7"),
#                                       result_file=path_results,
#                                       dataset=d[0], num_classes=d[1],
#                                       batch_size=512,
#                                       attack=a, eot_samples=e,
#                                       noise=n,
#                                       batch_prediction=5, sigma=0.25)
#                 print(job.job_id)


# #################################################################### AT models #####################################################################################
# datasets = [("CIFAR10", 10), ("CIFAR100", 100)]
# adv_tr = ["PGDLinf", "PGDL2", "CW"]

# # no noise at inference
# for a in attacks:
#     for d in datasets:
#         for at in adv_tr:

#             path_models = "models_copy/models"
#             path_logs = "logs_attacks"
#             path_results = "results_attacks"

#             path_models = os.path.join(path_models, d[0])
#             path_logs = os.path.join(path_logs, d[0])
#             path_results = os.path.join(path_results, d[0])

#             if not os.path.exists(path_results):
#                 os.makedirs(path_results)

#             path_models = os.path.join(path_models, "models_AT", at)
#             path_logs = os.path.join(path_logs, "models_AT", at)
#             path_results = os.path.join(path_results, "models_AT", at)

#             if not os.path.exists(path_results):
#                 os.makedirs(path_results)

#             result_file = "attacks"
#             path_results = os.path.join(path_results, result_file)

#             if os.path.exists(path_results+".txt"):
#                 delete_line(path_results+".txt", a)
#             # submission interface (logs are dumped in the folder)
#             executor = submitit.AutoExecutor(folder=path_logs)  # "uninterrupted")  # timeout in min
#             gpus_per_node = 8
#             executor.update_parameters(nodes=1, gpus_per_node=gpus_per_node, tasks_per_node=gpus_per_node,
#                                        cpus_per_task=int(80/gpus_per_node), timeout_min=4320,
#                                        partition="priority", comment="Deadline ACML 15/07", mem_gb=256, constraint="volta32gb")
#             init_file = get_init_file().as_uri()

#             job = executor.submit(main, init_file=init_file, path_model=os.path.join(path_models, "BEST.t7"),
#                                   result_file=path_results,
#                                   dataset=d[0], num_classes=d[1],
#                                   batch_size=256,
#                                   attack=a, eot_samples=1,
#                                   noise=None,
#                                   batch_prediction=1, sigma=0.25)
#             print(job.job_id)

#################################################################### RAT models #####################################################################################
datasets = [("CIFAR10", 10), ("CIFAR100", 100)]
adv_tr = ["PGDLinf", "PGDL2", "PGDL1"]
noises = ["Laplace", "Normal"]
eot_samples = [80]
# no noise at inference
for a in attacks:
    for e in eot_samples:
        for d in datasets:
            for at in adv_tr:
                for n in noises:

                    path_models = "models_copy/models"
                    path_logs = "logs_attacks"
                    path_results = "results_attacks"

                    path_models = os.path.join(path_models, d[0])
                    path_logs = os.path.join(path_logs, d[0])
                    path_results = os.path.join(path_results, d[0])

                    if os.path.exists(path_results+".txt"):
                        delete_line(path_results, [a, str(e)])

                    path_models = os.path.join(path_models, "models_RAT", n+"_"+at)
                    path_logs = os.path.join(path_logs, "models_RAT", n+"_"+at)
                    path_results = os.path.join(path_results, "models_RAT", n+"_"+at)

                    if not os.path.exists(path_results):
                        os.makedirs(path_results)

                    result_file = "attacks"
                    path_results = os.path.join(path_results, result_file)
                    if os.path.exists(path_results+".txt"):
                        delete_line(path_results+".txt", [a, str(e)])
                    # submission interface (logs are dumped in the folder)
                    executor = submitit.AutoExecutor(folder=path_logs)  # "uninterrupted")  # timeout in min

                    gpus_per_node = 8
                    executor.update_parameters(nodes=1, gpus_per_node=gpus_per_node, tasks_per_node=gpus_per_node,
                                               cpus_per_task=int(80/gpus_per_node), timeout_min=4320,
                                               partition="priority", comment="Deadline ACML 15/07", mem_gb=256, constraint="volta32gb")

                    init_file = get_init_file().as_uri()

                    job = executor.submit(main, init_file=init_file, path_model=os.path.join(path_models, "BEST.t7"),
                                          result_file=path_results,
                                          dataset=d[0], num_classes=d[1],
                                          batch_size=256,
                                          attack=a, eot_samples=e,
                                          noise=n,
                                          batch_prediction=5, sigma=0.25)
                    print(job.job_id)


#################################################################### MAT models #####################################################################################
# datasets = [("CIFAR10", 10), ("CIFAR100", 100)]
# mix_strategy = ["MixMean", "MixMax", "MixRand"]
# eot_samples = [80]
# # no noise at inference
# for a in attacks:
#     for d in datasets:
#         for s in mix_strategy:

#             path_models = "models_copy/models"
#             path_logs = "logs_attacks"
#             path_results = "results"

#             path_models = os.path.join(path_models, d[0])
#             path_logs = os.path.join(path_logs, d[0])
#             path_results = os.path.join(path_results, d[0])

#             if not os.path.exists(path_results):
#                 os.makedirs(path_results)

#             path_models = os.path.join(path_models, "models_MAT", s)
#             path_logs = os.path.join(path_logs, "models_MAT", s)
#             path_results = os.path.join(path_results, "models_MAT", s)

#             if not os.path.exists(path_results):
#                 os.makedirs(path_results)

#             result_file = "attacks"
#             path_results = os.path.join(path_results, result_file)
#             if os.path.exists(path_results+".txt"):
#                 delete_line(path_results+".txt", a)
#             # submission interface (logs are dumped in the folder)
#             executor = submitit.AutoExecutor(folder=path_logs)  # "uninterrupted")  # timeout in min
#             gpus_per_node = 8
#             executor.update_parameters(nodes=1, gpus_per_node=gpus_per_node, tasks_per_node=gpus_per_node,
#                                        cpus_per_task=int(80/gpus_per_node), timeout_min=4320,
#                                        partition="priority", comment="Deadline ACML 15/07", mem_gb=256, constraint="volta32gb")
#             init_file = get_init_file().as_uri()

#             job = executor.submit(main, init_file=init_file, path_model=os.path.join(path_models, "BEST.t7"),
#                                   result_file=path_results,
#                                   dataset=d[0], num_classes=d[1],
#                                   batch_size=256,
#                                   attack=a, eot_samples=e,
#                                   noise=None,
#                                   batch_prediction=1, sigma=0.25)
#             print(job.job_id)


# #################################################################### RMAT models #####################################################################################
# datasets = [("CIFAR10", 10), ("CIFAR100", 100)]
# mix_strategy = ["MixMean", "MixMax", "MixRand"]
# noises = ["Laplace", "Normal"]
# eot_samples = [80]
# # no noise at inference
# for a in attacks:
#     for d in datasets:
#         for s in mix_strategy:
#             for n in noises:

#                 path_models = "models_copy/models"
#                 path_logs = "logs_attacks"
#                 path_results = "results_attacks"

#                 path_models = os.path.join(path_models, d[0])
#                 path_logs = os.path.join(path_logs, d[0])
#                 path_results = os.path.join(path_results, d[0])

#                 if not os.path.exists(path_results):
#                     os.makedirs(path_results)

#                 path_models = os.path.join(path_models, "models_RMAT", n+'_'+s)
#                 path_logs = os.path.join(path_logs, "models_RMAT", n+'_'+s)
#                 path_results = os.path.join(path_results, "models_RMAT", n+'_'+s)

#                 if not os.path.exists(path_results):
#                     os.makedirs(path_results)

#                 result_file = "attacks"
#                 path_results = os.path.join(path_results, result_file)
#                 if os.path.exists(path_results+".txt"):
#                     delete_line(path_results+".txt", [a, str(e)])
#                 # submission interface (logs are dumped in the folder)
#                 executor = submitit.AutoExecutor(folder=path_logs)  # "uninterrupted")  # timeout in min
#                 gpus_per_node = 8
#                 executor.update_parameters(nodes=1, gpus_per_node=gpus_per_node, tasks_per_node=gpus_per_node,
#                                            cpus_per_task=int(80/gpus_per_node), timeout_min=4320,
#                                            partition="priority", comment="Deadline ACML 15/07", mem_gb=256, constraint="volta32gb")
#                 init_file = get_init_file().as_uri()

#                 job = executor.submit(main, init_file=init_file, path_model=os.path.join(path_models, "BEST.t7"),
#                                       result_file=path_results,
#                                       dataset=d[0], num_classes=d[1],
#                                       batch_size=256,
#                                       attack=a, eot_samples=e,
#                                       noise=n,
#                                       batch_prediction=5, sigma=0.25)
#                 print(job.job_id)
