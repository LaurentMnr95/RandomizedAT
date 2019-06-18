import submitit
from train_classifier import main
import os

datasets = [("CIFAR10",10),("CIFAR100",100)]
noises = ["Normal"] #[None,"Laplace","Normal"]

#Baseline+RT
for d in datasets:
	for n in noises:
		path_models = "models"
		path_logs = "logs"
		path_models=os.path.join(path_models,d[0])
		path_logs=os.path.join(path_logs,d[0])
		if n is None:
			path_models=os.path.join(path_models,"baseline")
			path_logs=os.path.join(path_logs,"baseline")

		else:
			path_models=os.path.join(path_models,"models_RT",n)
			path_logs=os.path.join(path_logs,"models_RT",n)
		executor = submitit.AutoExecutor(folder=path_logs)  # submission interface (logs are dumped in the folder)
		executor.update_parameters(gpus_per_node=2, timeout_min=4320, partition="dev")#"uninterrupted")  # timeout in min
		job = executor.submit(main,path_model=path_models,
		            dataset=d[0],num_classes = d[1],
		            epochs=200,batch_size=256,
		            resume_epoch=0, save_frequency=2,
		            adversarial_training=None,
		            noise=n,sigma=0.25)  # will compute add(5, 7)
		print(job.job_id)  # ID of your job


# AT 
# import submitit
# from train_classifier import main
# import os

# datasets = [("CIFAR10",10),("CIFAR100",100)]
# attacks = ["CW","PGDinf","PGDL2"]


# for d in datasets:
# 	for a in attacks:
# 		path_models = "models"
# 		path_logs = "logs"
# 		path_models=os.path.join(path_models,d[0])
# 		path_logs=os.path.join(path_logs,d[0])
# 		if n is None:
# 			path_models=os.path.join(path_models,"baseline")
# 			path_logs=os.path.join(path_logs,"baseline")

# 		else:
# 			path_models=os.path.join(path_models,"models_AT",a)
# 			path_logs=os.path.join(path_logs,"models_AT",a)
# 		executor = submitit.AutoExecutor(folder=path_logs)  # submission interface (logs are dumped in the folder)
# 		executor.update_parameters(gpus_per_node=8, timeout_min=4320, partition="uninterrupted")#"uninterrupted")  # timeout in min
# 		job = executor.submit(main,path_model=path_models,
# 		            dataset=d[0],num_classes = d[1],
# 		            epochs=200,batch_size=256,
# 		            resume_epoch=0, save_frequency=2,
# 		            adversarial_training=a,
# 		            noise=None,sigma=0.)  # will compute add(5, 7)
# 		print(job.job_id)  # ID of your job


