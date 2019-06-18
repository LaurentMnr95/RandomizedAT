import submitit
from test_classifier import main
import os

datasets = [("CIFAR10",10),("CIFAR100",100)]
# TODO add rm result file 
# no noise at inference RT
for d in datasets:
	path_models = "models"
	path_logs = "logs_test"
	path_results = "results"

	path_models=os.path.join(path_models,d[0])
	path_logs=os.path.join(path_logs,d[0])
	path_results=os.path.join(path_results,d[0])
	
	path_models=os.path.join(path_models,"baseline")
	path_logs=os.path.join(path_logs,"baseline")
	path_results=os.path.join(path_results,"baseline")

	if not os.path.exists(path_results):
		os.makedirs(path_results)

	executor = submitit.AutoExecutor(folder=path_logs)  # submission interface (logs are dumped in the folder)
	executor.update_parameters(gpus_per_node=2, timeout_min=4320, partition="dev")#"uninterrupted")  # timeout in min
	result_file = "no_attack.txt"
	func = submitit.helpers.FunctionSequence(verbose=True)
	ls_models = os.listdir(path_models)

	for m in ls_models:
		func.add(main,path_model=os.path.join(path_models,m),
					result_file=os.path.join(path_results,result_file),
					dataset="CIFAR10",num_classes = 10,
		            batch_size=256,
		            attack="No", batch_eot=1,
		            noise="No",
		            batch_prediction=3, sigma=0.25)
	executor.submit(func) # will compute add(5, 7)


datasets = [("CIFAR10",10),("CIFAR100",100)]
noises = ["Laplace","Normal"]
batchs_prediction = [1,5,10,20,50]
# TODO add rm result file 
for d in datasets:
	for n in noises:
		for b in batchs_prediction:
			path_models = "models"
			path_logs = "logs_test"
			path_results = "results"

			path_models=os.path.join(path_models,d[0])
			path_logs=os.path.join(path_logs,d[0])
			path_results=os.path.join(path_results,d[0])


			path_models=os.path.join(path_models,"models_RT",n)
			path_logs=os.path.join(path_logs,"models_RT",n)
			path_results=os.path.join(path_results,"models_RT",n)
			if not os.path.exists(path_results):
				os.makedirs(path_results)

			executor = submitit.AutoExecutor(folder=path_logs)  # submission interface (logs are dumped in the folder)
			executor.update_parameters(gpus_per_node=2, timeout_min=4320, partition="dev")#"uninterrupted")  # timeout in min
			result_file = "no_attack.txt".format(b)
			func = submitit.helpers.FunctionSequence(verbose=True)
			ls_models = os.listdir(path_models)

			for m in ls_models:
				func.add(main,path_model=os.path.join(path_models,m),
							result_file=os.path.join(path_results,result_file),
							dataset="CIFAR10",num_classes = 10,
				            batch_size=256,
				            attack="No", batch_eot=1,
				            noise=n, batch_prediction=b, sigma=0.25)
			executor.submit(func) # will compute add(5, 7)




datasets = [("CIFAR10",10),("CIFAR100",100)]
attacks = ["PGDinf","PGDL2","CW"]

# no noise at inference 
for d in datasets:
	for a in attacks:
		path_models = "models"
		path_logs = "logs_test"
		path_results = "results"

		path_models=os.path.join(path_models,d[0])
		path_logs=os.path.join(path_logs,d[0])
		path_results=os.path.join(path_results,d[0])

		if not os.path.exists(path_results):
			os.makedirs(path_results)


		path_models=os.path.join(path_models,"models_AT",a)
		path_logs=os.path.join(path_logs,"models_AT",a)
		path_results=os.path.join(path_results,"models_AT",a)
		if not os.path.exists(path_results):
			os.makedirs(path_results)

		executor = submitit.AutoExecutor(folder=path_logs)  # submission interface (logs are dumped in the folder)
		executor.update_parameters(gpus_per_node=2, timeout_min=4320, partition="dev")#"uninterrupted")  # timeout in min
		result_file = "no_attack.txt"
		
		func = submitit.helpers.FunctionSequence(verbose=True)
		ls_models = os.listdir(path_models)

		for m in ls_models:
			func.add(main,path_model=os.path.join(path_models,m),
						result_file=os.path.join(path_results,result_file),
						dataset=d[0],num_classes = d[1],
						batch_size=256,
						attack="No", batch_eot=1,
						noise="No",sigma=0)
		executor.submit(func) # will compute add(5, 7)

