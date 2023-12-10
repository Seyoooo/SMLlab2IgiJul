# Polish Careless Whisper
**Course:** Scalable Machine Learning and Deep Learning - Lab 1
**Team**: Iga Pawlak, Julien Horvat

A lab exercise to build a Machine Learning system for speech transcription in Polish. 
## 1. System components
The [Colab notebook](https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/fine_tune_whisper.ipynb) used as a baseline has in the first phase been split into feature and training pipelines. 

### 1.1 Feature pipeline
Following the given google collab code, this notebook import the dataset polish dataset from huggingface, resample and compute the log-Mel spectrogram of the input features, and save the processed features in google Drive. Note that in order to keep enough space on google drive for checkpointing, we used only 50% of training, 50% of validation, and 10% of test polish datasets. This notebook is running on a CPU.  

### 1.2 Training pipeline
This notebook is fine-tuning the pretrained whisper small model developped and trained by open AI, and available on Huggingface. After loading the pre-processed dataset from google drive, we check if we can import a pretrained model, and then set up the training. As google colab is disconnecting you after a few hours (sometimes less than one), we set the checkpoint save step at 100 steps. We trained on 4000 steps and saved checkpoints on google drive. Training took more than 10 hours. This notebook is running on a GPU with at least 15GB.

### 1.3 User interface 
The UI has been created in the form of an [app](https://huggingface.co/spaces/PiJul/PolishCarelessWhisper) on HuggingFace. 

There are a few tabs allowing the user to transcribe speech recorded from their microphone, as well as paste a link to a video on YouTube 

We also added the option to simply search a phrase on YouTube and choose how many seconds should be transcribed. 

## 2. Possible improvements
### (a) Model centric approach
An obvious improvement could be that choosing the bigger versions of Whisper could improve the quality of the transcription. Also hyperparameters such as `learning_rate`, `batch_size`, `gradient_accumulation_steps` could be manipulated to see which value provides better results. Also, set precision to 32 instead of 16 (`fp16=True`) could slightly improve performance, but would be costly in terms of model size. 

We could also modify the parameters of existing regularisation techniques such as dropout, label smoothing or level normalisation, introduced in the original [paper](https://arxiv.org/abs/1706.03762). Another possibility would be to introduce new regularisation techniques such as [DropDim](https://arxiv.org/pdf/2304.10321.pdf) where instead of disabling neurons, we hide certain dimensions of the embedding, forcing the network to produce features encodings based on incomplete semantic information. 

### (b) Data centric approach

There are existing datasets with a bigger number of samples such as [this one](https://doi.org/10.35111/twqh-f096). There are also domain-specific datasets that might prove very useful for fine-tuning transcription for a task with very specialised language, such as medical applications (example [dataset](https://www.futurebeeai.com/dataset/monologue-speech-dataset/healthcare-scripted-speech-monologues-polish-poland)). 
Some data augmentation techniques could also be used to increase the dataset. This [blog](https://towardsdatascience.com/data-augmentation-for-speech-recognition-e7c607482e78) introduce some for speech recognition. 
