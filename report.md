# Unsupervised Detection of Anomalous Sounds for Machine Condition Monitoring

## Introduction (Samuel)
- anomaly detection
- dataset
- baseline model

## Methodology (Teemu)

Inspired by the usage of convolutional autoencoders in image generation and noise removal [A], we implemented a convolutional autoencoder to reconstruct audio samples similarly as the baseline implementation. Usually image generation solutions are using variational autoencoders, but in outlier detection setup, controllable latent space do not provide any extra capabilities. So we deciced to stick with simple convolutional autoencoders that has convolutional layers with stride in encoder to reduce the size of the image and transpose convolution with stride in decoder to upsample the size back to original. We adopted the use of mean squared error as loss function in training and also in calculation of anomaly score. A brief experiments with binary crossentropy combined with KL-divergence inspired by [A] were made as loss function and measure of outlier with poor improvements to results.  

We considered mel-spectrograms as input images to our network. The dimensionality of the spectrograms was 64x64 and to get the 10 s input audio with sampling rate of 16 kHz clips transformed to spectrograms we therefore used $n_{mel}=64, n_{fft}=4096 $ and $hop\_lenght=2500$. The reduction of information in temporal domain is significant due to rather large $n_{fft}$ and hop_lenght but it usually can present the outliers of the test data well as presented in conclusions section. We also normalize the spectrograms to zero mean and variance of 1 to regularize the outputs of layers and get better drop the initial loss. Naturally, the channel number in spectrograms in 1 so the input size to network is (64,64,1).

The specific architecture of our network is as follows:

| Layer Type         |	Output Shape   | Num Parameters |
|--------------------|-----------------|----------------|
| Conv2D             | (32, 32, 32)    |	320  |
| BatchNormalization | (32, 32, 32)    |	128  |
| Conv2D             | (16, 16, 64)    |	18496|
| BatchNormalization | (16, 16, 64)    |	256  |
| Conv2D             | (8, 8, 64)      |	36928|
| BatchNormalization | (8, 8, 64)      |	256  |
| Conv2D             | (4, 4, 128)     |	73856|
| BatchNormalization | (4, 4, 128)     |	512  |
| Conv2DTranspose    | (8, 8, 64)      |	73792|
| BatchNormalization | (8, 8, 64)      |	256  |
| Conv2DTranspose    | (16, 16, 64)    |	36928|
| BatchNormalization | (16, 16, 64)    |	256  |
| Conv2DTranspose    | (32, 32, 32)    |	18464|
| BatchNormalization | (32, 32, 32)    |	128  |
| Conv2DTranspose    | (64,64, 1)      |	289  |
| BatchNormalization | (64,64, 1)      |	4    |

We implemented a kernel size of 3 to all convolutional and transpose convolutional layers and use stride 2 with padding='same' and Rectified Linear as activation across all layers. In total the model has 259 971 trainable parameters with is similar with the baseline implementation of the dcase task authors [X]. Note that parameters of the batch normalization layers are not count as trainable parameters because those are not optimized via gradient descend. 

As observable from table we adopted a symmetric architecture to our autoencoder where channel number and therefore also the number of parameters is similar in encoder and decoder parts of the network, but in reverse order. We noticed that the symmetry is essential to get the network to learn anything.

## Experiments and results (Juha)

## Conclusion (Juha, Teemu, Samuel)

Here are some original and reconstructed spectrograms from our models. 

![comparison of spectrograms for pump](pump_00_anomaly.png)

![comparison of spectrograms for Toy Car](toycar_anomaly_org.png)

Figure 1: Original mel-spectrogram on left and reconstruction of model in right of anomalious sample. First row is pump with id 0 and second row Toycar with id 3.

As we can see the models have learned to reconstruct also the anomalous samples even thought those are unseen in training. This yields poor results because the normal and anomalous samples can't be differentiated based on mean squared error. 

One reason that might cause the model to learn also to reconstruct the anomalious samples is that the convolutional models are more effective to learn the local structures in spectrogram. This leads to model that is capable of producing overall shapes that are unseen based on the local features. The independent features of anomalies like vertical lines or bright dots are still visible in some training data samples where the convolutional kernels are able to learn to construct those.

## References
Authors asked to cite to these in dcase page:


[Z] Yuma Koizumi, Shoichiro Saito, Hisashi Uematsu, Noboru Harada, and Keisuke Imoto. ToyADMOS: a dataset of miniature-machine operating sounds for anomalous sound detection. In Proceedings of IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), 308–312. November 2019. URL: https://ieeexplore.ieee.org/document/8937164.

[Y] Harsh Purohit, Ryo Tanabe, Takeshi Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi. MIMII Dataset: sound dataset for malfunctioning industrial machine investigation and inspection. In Proceedings of the Detection and Classification of Acoustic Scenes and Events 2019 Workshop (DCASE2019), 209–213. November 2019. URL: http://dcase.community/documents/workshop2019/proceedings/DCASE2019Workshop_Purohit_21.pdf.

[X] Yuma Koizumi, Yohei Kawaguchi, Keisuke Imoto, Toshiki Nakamura, Yuki Nikaido, Ryo Tanabe, Harsh Purohit, Kaori Suefusa, Takashi Endo, Masahiro Yasuda, and Noboru Harada. Description and discussion on DCASE2020 challenge task2: unsupervised anomalous sound detection for machine condition monitoring. In Proceedings of the Detection and Classification of Acoustic Scenes and Events 2020 Workshop (DCASE2020), 81–85. November 2020. URL: http://dcase.community/documents/workshop2020/proceedings/DCASE2020Workshop_Koizumi_3.pdf.

[A] Francois Chollet, Building Autoencoders in Keras, 2016, https://blog.keras.io/building-autoencoders-in-keras.html
