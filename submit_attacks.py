import submitit
from test_classifier import main
import os
from utils import delete_line

attacks = [None, "CW", "EAD", "PGDLinf", "PGDL2", "PGDL1"]
#################################################################### Baseline models #####################################################################################
datasets = [("CIFAR10", 10), ("CIFAR100", 100)]
# TODO add rm result file
# no noise at inference baseline
for d in datasets:
    for a in attacks:
        path_models = "models"
        path_logs = "logs_attacks"
        path_results = "results"

        path_models = os.path.join(path_models, d[0])
        path_logs = os.path.join(path_logs, d[0])
        path_results = os.path.join(path_results, d[0])

        path_models = os.path.join(path_models, "baseline")
        path_logs = os.path.join(path_logs, "baseline")
        path_results = os.path.join(path_results, "baseline")

        if not os.path.exists(path_results):
            os.makedirs(path_results)

        result_file = "attacks.txt"
        path_results = os.path.join(path_results, result_file)

        delete_line(path_results, a)

        # submission interface (logs are dumped in the folder)
        executor = submitit.AutoExecutor(folder=path_logs)  # "uninterrupted")  # timeout in min
        executor.update_parameters(
            gpus_per_node=2, timeout_min=4320, partition="dev")
        job = executor.submit(main, path_model=os.path.join(path_models, "BEST.t7"),
                              result_file=path_results,
                              dataset=d[0], num_classes=d[1],
                              batch_size=256,
                              attack=a, eot_samples=1,
                              noise=None,
                              batch_prediction=1, sigma=0.25)
        print(job.job_id)

#################################################################### RT models #####################################################################################
# datasets = [("CIFAR10", 10), ("CIFAR100", 100)]
# noises = ["Laplace", "Normal"]
# for d in datasets:
#     for n in noises:
#         path_models = "models"
#         path_logs = "logs_test"
#         path_results = "results"

#         path_models = os.path.join(path_models, d[0])
#         path_logs = os.path.join(path_logs, d[0])
#         path_results = os.path.join(path_results, d[0])

#         path_models = os.path.join(path_models, "models_RT", n)
#         path_logs = os.path.join(path_logs, "models_RT", n)
#         path_results = os.path.join(path_results, "models_RT", n)

#         if not os.path.exists(path_results):
#             os.makedirs(path_results)

#         result_file = "noise_eval.txt"
#         path_results = os.path.join(path_results, result_file)

#         if os.path.exists(path_results):
#             os.remove(path_results)

#         # submission interface (logs are dumped in the folder)
#         executor = submitit.AutoExecutor(folder=path_logs)
#         # "uninterrupted")  # timeout in min
#         executor.update_parameters(
#             gpus_per_node=4, timeout_min=4320, partition="dev")

#         func = submitit.helpers.FunctionSequence(verbose=True)
#         ls_models = os.listdir(path_models)

#         for m in ls_models:
#             func.add(main, path_model=os.path.join(path_models, m),
#                      result_file=path_results,
#                      dataset=d[0], num_classes=d[1],
#                      batch_size=256,
#                      attack=None, eot_samples=1,
#                      noise=n, batch_prediction=5, sigma=0.25)
#         executor.submit(func)  # will compute add(5, 7)


#################################################################### AT models #####################################################################################
datasets = [("CIFAR10", 10), ("CIFAR100", 100)]
adv_tr = ["PGDLinf", "PGDL2", "CW"]

# no noise at inference
for d in datasets:
    for at in adv_tr:
        for a in attacks:
            path_models = "models"
            path_logs = "logs_test"
            path_results = "results"

            path_models = os.path.join(path_models, d[0])
            path_logs = os.path.join(path_logs, d[0])
            path_results = os.path.join(path_results, d[0])

            if not os.path.exists(path_results):
                os.makedirs(path_results)

            path_models = os.path.join(path_models, "models_AT", at)
            path_logs = os.path.join(path_logs, "models_AT", at)
            path_results = os.path.join(path_results, "models_AT", at)

            if not os.path.exists(path_results):
                os.makedirs(path_results)

            result_file = "attacks.txt"
            path_results = os.path.join(path_results, result_file)
            delete_line(path_results, a)
            # submission interface (logs are dumped in the folder)
            executor = submitit.AutoExecutor(folder=path_logs)  # "uninterrupted")  # timeout in min
            executor.update_parameters(
                gpus_per_node=2, timeout_min=4320, partition="dev")
            job = executor.submit(main, path_model=os.path.join(path_models, "BEST.t7"),
                                  result_file=path_results,
                                  dataset=d[0], num_classes=d[1],
                                  batch_size=256,
                                  attack=a, eot_samples=1,
                                  noise=None,
                                  batch_prediction=1, sigma=0.25)
            print(job.job_id)
