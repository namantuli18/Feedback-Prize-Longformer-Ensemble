# Feedback-Prize-Longformer-Ensemble

## Results
| Public Score | Private Score | Public Rank | Private Rank |
|----------|----------|----------|----------|
| 0.705 | 0.716 | 70/2000  | 52/2000

## Problem Statement  
The participants in this competition were asked to segment essays of students in 6th–12th grade in one of 15 categories, hoping to enhance the feedback process and help students improve their writing skills.

## Dataset
The textual data for the competition was divided into 2 separate files, i.e., **train.zip** and **test.zip**, which consisted of the training and testing sets of documents, respectively. The **train.zip** file had roughly ~15K .txt files, with their respective annotations present in the **train.csv** file. SImilarly, the test set consisted of ~10K unseen documents, that were required to be segmented into various discourse elements such as:
* Lead - an introduction that begins with a statistic, a quotation, a description, or some other device to grab the reader’s attention and point toward the thesis
* Position - an opinion or conclusion on the main question
* Claim - a claim that supports the position
* Counterclaim - a claim that refutes another claim or gives an opposing reason to the position
* Rebuttal - a claim that refutes a counterclaim
* Evidence - ideas or examples that support claims, counterclaims, or rebuttals.
* Concluding Statement - a concluding statement that restates the claims 
<img src="https://github.com/namantuli18/Feedback-Prize-Longformer-Ensemble/blob/main/imgs/dataset.png" width="600" height="300" />

### Resources to the dataset:  
Aigner Picou, Alex Franklin, Maggie, Meg Benner, Perpetual Baffour, Phil Culliton, Ryan Holbrook, Scott Crossley, Terry_yutian, ulrichboser. (2021). Feedback Prize - Evaluating Student Writing. Kaggle. https://kaggle.com/competitions/feedback-prize-2021

## Evaluation Metric  
Submissions are evaluated on the overlap between ground truth and predicted word indices.

1. For each sample, all ground truths and predictions for a given class are compared.
2. If the overlap between the ground truth and prediction is >= 0.5, and the overlap between the prediction and the ground truth >= 0.5, the prediction is a match and considered a true positive. If multiple matches exist, the match with the highest pair of overlaps is taken.
3. Any unmatched ground truths are false negatives and any unmatched predictions are false positives.

## Methodology

### Curating the dataset
1. Since the textual data is written by students, it was important to correct the grammatical errors (if any) before training the models.
2. Thus, we corrected any basic spelling mistakes using the `Speller` package in Python, segregating the data into 5 separate folds for training and aiming to stabilize the cross-validation score.
3. However, the testing dataset could also have certain grammatical errors since it was written by students. Spelling mistakes were also corrected in the testing dataset.
4. Overall, our final strategy incorporated models that were trained on both samples, i.e., with and without spelling mistakes, in order to devise a more comprehensive algorithm.

### Model Training

Given the large sequence length of each training sample, we decided to use [Longformers](https://huggingface.co/docs/transformers/model_doc/longformer) because of its self-attention mechanism and the ability to process large records of text within optimal inference time. For text segmentation tasks, longformers usually outperform traditional algorithms because of the following characteristics:
- Global & Local attention capturing and retention
- Increased efficiency because of sliding window attention
- Ability to incorporate lengthy textual information

Pytorch and hugging face transformers were used for training our models over the 5 training folds on a Google Colab Pro machine. The training process roughly took around 10 hours for 5 epochs over each fold.
Although we tuned parameters to infuse variance among the final result, the default training parameters have been tabulated below:
| Parameter            | Value                                            |
|----------------------|--------------------------------------------------|
| loss                 | [cross entropy](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
| learning_rate        | 1e-5                                             |
| epochs               | 5                                                |
| batch_size           | 4                                                |
| valid_batch_size     | 4                                                |
| max_len              | 1536                                             |
| accumulation_steps   | 1                                                |

Although the default loss function was mostly used, we also tried using the following loss functions while training some of our models to have variability in the final evaluation results and avoid convergence:
- [Smooth Loss](https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html)
- [SCE Loss](https://github.com/HanxunH/SCELoss-Reproduce)
- [Focal Loss](https://github.com/clcarwin/focal_loss_pytorch)

### Model Evaluation and Inference

While evaluating model performance, great attention was paid to the correlation between the cross-validation (CV) and leaderboard (LB) scores in order to smoothen the score. Our approach tried to ensemble multiple algorithms, aiming to enhance the individual performance of the models.

The performance of our individual models, along with their weightage in our final ensembles, has been encapsulated below:
| S No | Model Name | Loss Function | Huggingface Link | Weightage in Ensemble |
|----------|----------|----------|----------|----------|
| 1. | Longformer Large | Cross Entropy Loss | [longformer-large-4096](https://huggingface.co/allenai/longformer-large-4096)| 0.1
| 2. | Longformer Trivia | Cross Entropy Loss | [longformer-large-4096-finetuned-triviaqa](https://huggingface.co/allenai/longformer-large-4096-finetuned-triviaqa)| 0.1
| 3. | Deberta XLarge | Cross Entropy Loss | [ddeberta-v2-xlarge](https://huggingface.co/microsoft/deberta-v2-xlarge)| 0.3
| 4. | Funnel Tranformer | Cross Entropy Loss | [funnel-transformer](https://huggingface.co/docs/transformers/model_doc/funnel)| 0.2
| 5. | Deberta Large | Smooth Loss | [deberta-large](https://huggingface.co/microsoft/deberta-large)| 0.3

Algorithm-wise CV/LB trends have been shown in the below image:  
<img src="https://github.com/namantuli18/Feedback-Prize-Longformer-Ensemble/blob/main/imgs/cv-lb.png" width="500" height="300" />

**Note** : We also trained other models, such as the Deberta XLarge-mnli or the Birdformer variations, but they were not included in our final submissions because of considerations pertinent to their longer training & inference time or overfitting the public leaderboard.


## What did not work
1. Training a model over a single fold exhibited a reduced inference time but performed poorly as compared to models that were trained on a larger number of folds. 
2. Text augmentation over training data was not effective for us and led to a reduced CV when tried with Longformer. We did not try this for other models, though. 
3. Although it may have been slightly beneficial, the spell-checking did not significantly affect the performance of our models. 
4. Use of models from open-source libraries such as SpaCy

## What could have worked
1. We did not spend a lot of time filtering the dataset. The dataset could have been filtered based on the weightage of classes to remove bias concentrated towards one particular class. 
2. It was too late for us in the competition to incorporate pseudo-labeling, but it certainly worked for many competitors. 
3. Training more data-intensive versions of models like XLarge could have worked. With the constrained resources, it could not be implemented.

## Code 
For training code, you can refer file `scripts/train-longformer.ipynb`  
For inference script, please refer [Kaggle Notebook](https://www.kaggle.com/code/namantuli/clean-feedback-ensemble-balanced) or file `scripts/clean-feedback-ensemble-balanced.ipynb`




 


