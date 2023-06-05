# RLScheduler Using Pytorch
This repo includes the deep batch scheduler source code and necessary datasets to run the experiments/tests. 

The code has been tested on Ubuntu 18.04 with PyTorch 1.13 and Gym 0.21. 
## Citing RLScheduler
The relevant research paper has been published at SC20. If you reference or use RLScheduler in your research, please cite:

```
@inproceedings{zhang2020rlscheduler,
  title={RLScheduler: an automated HPC batch job scheduler using reinforcement learning},
  author={Zhang, Di and Dai, Dong and He, Youbiao and Bao, Forrest Sheng and Xie, Bing},
  booktitle={SC20: International Conference for High Performance Computing, Networking, Storage and Analysis},
  pages={1--15},
  year={2020},
  organization={IEEE}
}
```

## Installation
### Required Software
* Python 3.9 and PyTorch
Use VirtuanEnv or Conda to build a Python3.9 environment and PyTorch at least 1.13.0
Note that, we do not leverage GPUs, so no need to configure the GPU version of PyTorch.

* OpenMPI and mpi4py
```bash
sudo apt-get install openmpi-bin openmpi-doc libopenmpi-dev
conda install mpi4py
```

### Clone Deep Batch Scheduler
```bash
git clone https://github.com/daidong/rlscheduler-pytorch.git
```

### Install Dependencies
```shell script
cd rlscheduler-pytorch
pip install -r requirements.txt
```

### File Structure

```
data/: Contains a series of workload and real-world traces.
plot.py: Plot the trained results.
rlschedule.py: The main rlscheduler file.
schedgym.py: the SchedGym environment.
```

To change the hyper-parameters, such as `MAX_OBSV_SIZE` or the trajectory length during training, you can change them in rlschedule-torch.py.

### Training
To train a RL model based on a job trace, run this command:
```bash
python rlschedule.py --workload "./data/lublin_256.swf" --exp_name lublin_256 --trajs 100 --seed 0 --cpu 4
```

There are many other parameters in the source file.
* `--model`, specify a saved trained model (for two-step training and re-training)
* `--pre_trained`, specify whether this training will be a twp-step training or re-training
* `--score_type`, specify which scheduling metrics you are optimizing for: [0]：bounded job slowdown；[1]: job waiting time; [2]: job response time; [3] system resource utilization.
* `--cpu`, specify how many CPU cores you want to use to do the parallel training.

### Monitor Training 

After running Default Training, a folder named `logs/your-exp-name/` will be generated. 

```bash
python plot.py ./data/logs/your-exp-name/ -x Epoch -s 1
```

It will plot the training curve.

### Test and Compare

After RLScheduler converges, you can test the result and compare it with different policies such as FCFS, SJF, WFP3, UNICEP, and F1.

```bash
python compare-pick-jobs.py --rlmodel "./logs/your-exp-name/your-exp-name_s0/" --workload "./data/lublin_256.swf" --len 2048 --iter 10
```
There are many parameters you can use:
* `--seed`, the seed for random sampling
* `--iter`, how many iterations for the testing
* `--backfil`, enable/disable backfilling during the test
* `--score_type`, specify the scheduling metrics. [0]：bounded job slowdown；[1]: job waiting time; [2]: job response time; [3] system resource utilization.