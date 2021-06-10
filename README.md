# Sharing Less is More: Lifelong Learning in Deep Networks with Selective Layer Transfer
This code repository is for deep lifelong learning paper [LASEM at ICML 2021]().

Instead of designing new way of knowledge transfer between tasks, LASEM searches the optimal network structure (task-specific or knowledge-transfer at each layer) for the lifelong learning due to the empirical observation that layers to be shared is as critical as knowledge-transfer architecture for better performance. LASEM incorporated EM algorithm with lifelong learning architectures to figure out the optimal layers to be shared between tasks as well as optimal weights of the networks simultaneously. In this implementation, we used hard-parameter sharing, tensor-factorization and DF-CNN as base lifelong learner to apply LASEM.


## Version and Dependencies
- Python 3.6 or higher
- Tensorflow (1.14), psutil, scikit-image, scikit-learn, opencv-python

## Data
- CIFAR-100 (Lifelong)
    - Similar to CIFAR-10, but having 100 classes.
    - Each task is 10-class classification task, and there are 10 tasks for the lifelong learning task with heterogeneous task distribution (disjoint set of image classes for these sub-tasks).
    - We trained models by using only 4% of the available dataset.
    - We normalized images.

- Office-Home (Lifelong)
    - We used images in Product and Real-World domains.
    - Each task is 13-class classification task, and image classes of sub-tasks are randomly chosen without repetition (but distinguishing classes from Product domain and those from Real-World domain).
    - Images are rescaled to 128x128 size and rescaled range of pixel value to [0, 1], but not normalized or augmented.

- Office-Home Incremental (Lifelong)
    - We used images of Couch, Table, Paper Clip, Clipboards, Bucket, Notebook, Bed, Trash Can in Product domains to make 6 tasks.
    - First task is 3-class classification task (Couch, Table and Paper Clip), and image classes of each task is one new class on top of image classes of earlier task. Above list of classes are in the order of adding into task.
    
- STL-10 (Lifelong)
    - STL-10 dataset consists of images of 10 classes with resolution of 96x96.
    - We generated 20 tasks of three-way classification randomly. (random selection on image classes, mean and variance of Gaussian noise, and the order of channel permutation for the task)
    - We rescaled range of image value to [-0.5, 0.5].

## Proposed Model
- Hybrid DF-CNN model (LL_hybrid_DFCNN_minibatch model in the code)

- LASEM (Lifelong Architecture Search via EM) applied to hard-parameter sharing, tensor factorization and DF-CNN model (models in the code cnn_lasem_model.py)


## Baseline Model
- Single Task model
    - Construct independent models as many as the number of tasks, and train them independently.

- Single Neural Net model
    - Construct a single neural network, and treat data of all task as same.

- Hard-parameter Shared model
    - Neural networks for tasks share convolution layers, and have independent fully-connected layers for output.

- [Tensor Factorization](https://arxiv.org/abs/1605.06391) [model](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-BulatA.1460.pdf)
    - Factorize parameter of each layer into multiplication of several tensors, and share one tensor over different tasks as knowledge base. (Details in the paper Bulat, Adrian, Jean Kossaifi, Georgios Tzimiropoulos, and Maja Pantic. "Incremental multi-domain learning with network latent tensor factorization." ICML (2020).)

- [Dynamically Expandable Network model](https://arxiv.org/abs/1708.01547)
    - Extended hard-parameter shared model by retraining some neurons selectively/adding new neurons/splitting neurons into disjoint groups for different set of tasks according to the given data.
    - The code (cnn_den_model.py) is provided by authors which we do not have authority to release.
    
- [Progressive Neural Network](https://arxiv.org/abs/1606.04671)
    - Introduce layer-wise lateral connections from earlier tasks to the current task to allow one-way transfer of knowledge.
    
- [Differentiable Architecture Search (DARTS)](https://arxiv.org/abs/1806.09055)
    - Neural Architecture Search method which is applicable to selective layer transfer.
    
- [Conditional Channel Gated Network](https://arxiv.org/abs/2004.00070)

- [Additive Parameter Decomposition](https://openreview.net/forum?id=r1gdj2EKPB)

- [Deconvolutional-Factorized CNN (DF-CNN)](https://www.ijcai.org/proceedings/2019/393)
    - Factorize parameters of each layer into a shared tensor (a.k.a. knowledge base) and deconvolution-based task-specific mappings
    - The code (cnn_dfcnn_model.py) is based on the released code of the authors.


## Code Files
1. main.py
    main function to set-up model and data, and run training for classification tasks.

2. Directory *classification*
    This directory contains codes for classification tasks.
    - gen_data.py
        load dataset and convert its format for MTL/LL experiment
    - train_wrapper.py
        wrapper of train function in train.py to run independent training trials and save statistics into .mat file
    - train.py
        actual training function for DF-CNN and baselines exist.
    - Directory *model*
        This directory contains codes of neural network models.

3. Directory *utils*
    This directory contains utility functions.
    - utils_env_cl.py
        every hyper-parameter related to dataset and neural net models are defined.
    - utils.py
        functions which are miscellaneous but usable in general situation are defined. (e.g. handler of placeholder and output of MTL/LL models)
    - utils_nn.py and utils_tensor_factorization.py
        functions directly related to the construction of neural networks are defined. (e.g. function to generate convolutional layer)


## Sample command to train a specific model
1. Hard-parameter Sharing with Top-2 transfer configuration on CIFAR100

    ```python3 main.py --gpu 0 --data_type CIFAR100_10 --data_percent 4 --num_clayers 4 --model_type HPS --test_type 10 --lifelong --save_mat_name cifar100_hpsTop2.mat```

2. Tensor Factorization with Bottom-3 transfer configuration on Office-Home (Check utils/utils_env_cl.py for pre-defined transfer config)

    ```python3 main.py --gpu 0 --data_type OfficeHome --num_clayers 4 --model_type Hybrid_TF --test_type 30 --lifelong --save_mat_name officehome_tfBot3.mat```
    
3. Deconvolutional Factorized CNN with Alter. transfer configuration on STL-10

    ```python3 main.py --gpu 0 --data_type STL10_20t --num_clayers 6 --model_type Hybrid_DFCNN --test_type 55 --lifelong --save_mat_name stl10_dfcnnAlter.mat```

4. LASEM HPS/TF/DFCNN on CIFAR100 (change model_type argument to either LASEM_HPS/LASEM_TF/LASEM_DFCNN or LASEMG_HPS/LASEMG_TF/LASEMG_DFCNN for group-based LASEM)

    ```python3 main.py --gpu 0 --data_type CIFAR100_10 --data_percent 4 --num_clayers 4 --model_type LASEM_HPS --lifelong --save_mat_name cifar100_lasem_hps.mat```
    
5. Dynamically Expandable Net on Office-Home

    ```python3 main.py --gpu 0 --data_type OfficeHome --num_clayers 4 --model_type DEN --lifelong --save_mat_name officehome_den.mat```
    
6. Progressive Neural Net on Office-Home

    ```python3 main.py --gpu 0 --data_type OfficeHome --num_clayers 4 --model_type ProgNN --test_type 1 --lifelong --save_mat_name officehome_prognn.mat```
    
7. Differentiable Architecture Search (DARTS) HPS/DFCNN on Office-Home (change model_type argument to either DARTS_HPS/DARTS_DFCNN)

    ```python3 main.py --gpu 0 --data_type OfficeHome --num_clayers 4 --model_type DARTS_HPS --lifelong --save_mat_name officehome_darts_hps.mat```


## Citations
### LASEM / ICML 2021
```
Coming Soon!
```