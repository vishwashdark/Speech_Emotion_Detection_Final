# Speech_emotion_detection

METHODOLOGY
Speech processing is very necessary for many reasons and its application is all over the field of computer science and machine learning, Speech processing is important to understand the emotions of a person through the way they talk, try to understand the behavior of different voice patterns in this modern era as we see many voice activated alarms and bots which helps us in making our lives easy by automating a lot of tasks, in short speech processing is necessary because it helps us in human to computer reactions which enhances the accessibility, support various application in different industries.
  Librosa is a python library that is mainly designed for audio and music signal processing and provides a wide range of tools and functionalities that help in analyzing speech signals, it also helps in manipulation of audio signals and gather information about different wave signals. Librosa mainly focuses on the spectral analysis which us helps us in better understanding the speech signals. Librosa is also compatible to machine learning models they help us in preprocessing steps or other ml related tasks.
A.	Feature Extraction
1) Mel Frequency Cepstrum coefficient: MFCC is one of the most widely used feature extraction approach in speech processing domain. MFCC helps in representing spectral characteristics of audio signals in a compact form. MFCC helps in capturing the shape of the power spectrum of the audio signal. 

The input audio signal is first divided into short frames which is usually around 20 to 40 milliseconds and then windowing is applied to each frame in the form of Hamming or Hanning. The windowed frame is then converted into the frequency domain using various transformation technique usually Discrete Fourier Transform (DFT), after which the mel-filterbank is applied to audio, which represents the energy in different frequency band.
 	
The next step is to take logarithm of these energies, thereby compressing the range and making the features evenly distributed. The application of logarithm helps in bringing the audio more closely to the human auditory perception. After the application of logarithm scaling, transformation techniques are applied again to extract the most important and relevant features and transforms the filter bank energies to a set of cepstral coefficients.

2) Chroma features

Chroma features are calculates by mapping the frequency spectrum of the audio signal onto to the 12 pitch classes of chroma scale. Each pitch represents a specific musical note regardless of its octave. For example C3 and C4 will be mapped to the same pitch class C.
![image](https://github.com/vishwashdark/Speech_Emotion_Detection_Final/assets/92641662/ad8d3ae9-bf1e-40d7-84ec-f3f0d9c66421)

 
Figure 1: Chroma Features of the Dataset

The computation of chroma features as shown in Figure 1. is similar to that of the Mel Frequency, here the audio signal is transformed into the frequency domain by using Short term Fourier Transform (STFT). The chorma features are then found out using the spectrogram by averaging the spectral energy with each of the 12 pitches. 

The main advantage of chroma features is its constant behavior to the changes in the key or the pitch. By changing the keys for the same musical pattern will produce the same feature representation. They help in capturing essential musical characteristics while being robust to variations in pitch and key which makes it a very helpful tool in retrieval and analysis of audio signals.


2) Mel spectrogram features

The Mel scale in an intuitive scale of pitches which helps in resembling human auditory system’s response to different frequencies. It is formed based on the notion humans perceive changes in pitch logarithmically then linearly. As seen in the Figure 2. The mel scale is designed to impersonate the human ear’s frequency resolution, when the frequencies are lower the resolution is higher and the vice versa. The main component of Mel spectrogram features is the mel filterbank which is a set of triangular bandpass filters spaced evenly along the mel scale, where the peaks of the triangle are aligned to specific Mel frequencies and their bandwidths determined by adjacent Mel frequencies. Each triangular filter in the mel filterbank is created to capture the energy within a specific frequency range.

The computation of Mel spectrogram begins with the audio signal being divided into short overlapping frames. Each frame is then passed through the Mel filterbank and the energy within each filter’s passband is computed. The energies resulting from the filterbank is then logarithmically scaled to produce perceptually meaningful representation. The Mel spectrogram is a matric where each row corresponds to the different filter in the Mel filterbank and column corresponds to a time frame.

 ![image](https://github.com/vishwashdark/Speech_Emotion_Detection_Final/assets/92641662/6a685613-d401-41ab-9a32-76329e56a339)

Figure 2: Mel Frequency Features of the Dataset

B.	 Data Augmentation
1) Adding Noise: Adding noise is one of the most used augmentation methods, where the noise is introduced into the original audio signal by overlaying it with random noise. White noise has equal intensity across all frequencies, thereby is added to the audio signal.  The purpose of adding noise helps the model in getting exposed to noisy samples during training which is usually the real world scenario, so that the model learns to focus on relevant features of the audio.

1) Shifting: Shifting involves with disarranging the audio in time, thereby changing the starting point of the signal. Shifting is often applied to the spectrogram representation of the audio, where the frequency signal is analyzed over time. By bringing in random shifts in the data the model learns to recognize the same audio content even when it occurs at different time and also enhances the ability to recognize patterns in audio. 

C.	Label encoders
The process of converting labels that have categories into numerical representation is called label encoding which is taken care by label encoder. Since we are using the Ravdess dataset, the categorical representation represents different emotions present. The label encoder assigns unique integer to each category. The label encoding is followed by one hot encoding which converts the encoded label into binary matrix representations. This process helps in preparing the labels which makes the work of models easier. 

D.	Models 
The methodology consists the comparison between 4 different models for recognizing the emotion in the audio samples. 
Figure 3. as shown below, The Model Architecture include CNN, LSTM, Bi-LSTM, GRU and an ensemble of BI-LSTM and GRU.
 ![image](https://github.com/vishwashdark/Speech_Emotion_Detection_Final/assets/92641662/f21b370f-6b55-4ada-8371-fbf370d96fc8)

Figure 3: Model Architecture

1) Convolution Neural Networks (CNN): Convolutional neural networks (CNN) have always been extremely effective in capturing the structural patterns in the data, in terms of audio signal it helps in capturing the input spectrogram representation. The model starts with an input layer which has a sequence of length 180 with single feature dimension. The convolutional follows the input layer with 256 filters having kernel of size 5 and uses ReLu activation function. The convolution layer extracts features in this layer. 

The second convolutional layer of 128 filters and kernel size of 5 is introduced, here L1 and L2 regularization are applied to the weights, and this layer uses ReLu as the activation function. This layer focuses on extracting higher level features from the output of the first convolutional layer. The second Convolutional layer is followed with 2 more convolutional layers with the same configuration of that of the second one. There are two dropout layers which have dropout rates 0.1 and 0.5 respectively, the first is applied right after the first activation layer and a higher dropout is applied after the forth convolutional layer to regularize the model and reduce the risk of overfitting. The pooling layer has been used after the second convolutional layer, which helps in down sampling the feature maps obtained from the convolutional layers and retaining the most important features.  The flatten layer is used to convert the output to one-dimensional array and preparing it for fully connected layers. The dense layer perform classification of the emotions from the audio signals.

2) Long Short Term Memory (LSTM):To overcome the problems posed by both RNN and GRU we have used Long Short Term Memory (LSTM) along with CNN. LSTM plays a crucial role in capturing the long range dependencies in the temporal sequence of audio features which is extremely crucial for emotion recognition.
The model begins with the input layer which has dimension of (180,1). The input layer is followed by the LSTM layers. The first LSTM layer has 64 units, a dropout rate of 0.5 is applied to prevent overfitting which randomly drops 50 percent of units during training. The second LSTM layer also has the same configuration to that of the first one. The LSTM layer process the input and learns the temporal dependencies within the data. The dense layer is used to provide the probabilities for classifying the output classes using softmax activation function. 

3) Bidirectional Long Short-Term Memory (Bi-LSTM):	The input dimension is (180, 1), the model consists of two Bi-LSTM layers having 64 LSTM units, which return the full sequence of outputs, a dropout layer having dropout rate of 0.5 has been applied to prevent overfitting. The second LSTM layer also consists of 64 LSTM units which also consists of 0.5 droprate. The Bi-LSTM layers captures both past and future context in data hence the name bidirectional. This allows the models to learn the long term dependencies in the data. Dense layers have been used to provide the probabilities of the class which takes the output of BiLSTM layer and maps to the output class. 

4) Gated Recurrent Unit (GRU): The model consists of 2 GRU layers which have 64 GRU units each having dropout rates of 0.5 each which helps in preventing overfitting. The GRU layers process input sequence similar to that of LSTM to capture the context within the data. GRU units are much simpler than LSTM units which makes them computationally more efficient. The dense layers have been used to provide the probabilities of the class which takes the output of BiLSTM layer and maps to the output class. 

5) Ensemble BiLSTM and GRU:The architecture consists of two GRU layers with 64 units each and both having dropout rates as 0.5 to prevent overfitting. In this architecture the outputs of GRU are concatenated with the outputs of BiLSTM. The concatenated output serves as input to the dense layers which helps in producing the final classification probabilities. Ensemble models can perform better when compared to individual models by combining predictions of multiple base models. The ensemble mode are also less sensitive to noise and other outliers and can also capture different patterns present in the data.  
