{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: librosa in c:\\users\\satvi\\anaconda3\\lib\\site-packages (0.8.1)\n",
      "Requirement already satisfied: numpy>=1.15.0 in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from librosa) (1.19.2)\n",
      "Requirement already satisfied: numba>=0.43.0 in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from librosa) (0.51.2)\n",
      "Requirement already satisfied: decorator>=3.0.0 in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from librosa) (4.4.2)\n",
      "Requirement already satisfied: scikit-learn!=0.19.0,>=0.14.0 in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from librosa) (0.23.2)\n",
      "Requirement already satisfied: audioread>=2.0.0 in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from librosa) (2.1.9)\n",
      "Requirement already satisfied: packaging>=20.0 in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from librosa) (20.4)\n",
      "Requirement already satisfied: pooch>=1.0 in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from librosa) (1.5.2)\n",
      "Requirement already satisfied: scipy>=1.0.0 in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from librosa) (1.5.2)\n",
      "Requirement already satisfied: joblib>=0.14 in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from librosa) (0.17.0)\n",
      "Requirement already satisfied: resampy>=0.2.2 in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from librosa) (0.2.2)\n",
      "Requirement already satisfied: soundfile>=0.10.2 in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from librosa) (0.10.3.post1)\n",
      "Requirement already satisfied: llvmlite<0.35,>=0.34.0.dev0 in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from numba>=0.43.0->librosa) (0.34.0)\n",
      "Requirement already satisfied: setuptools in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from numba>=0.43.0->librosa) (50.3.1.post20201107)\n",
      "Requirement already satisfied: threadpoolctl>=2.0.0 in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from scikit-learn!=0.19.0,>=0.14.0->librosa) (2.1.0)\n",
      "Requirement already satisfied: six in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from packaging>=20.0->librosa) (1.15.0)\n",
      "Requirement already satisfied: pyparsing>=2.0.2 in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from packaging>=20.0->librosa) (2.4.7)\n",
      "Requirement already satisfied: appdirs in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from pooch>=1.0->librosa) (1.4.4)\n",
      "Requirement already satisfied: requests in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from pooch>=1.0->librosa) (2.24.0)\n",
      "Requirement already satisfied: cffi>=1.0 in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from soundfile>=0.10.2->librosa) (1.14.3)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from requests->pooch>=1.0->librosa) (2020.6.20)\n",
      "Requirement already satisfied: chardet<4,>=3.0.2 in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from requests->pooch>=1.0->librosa) (3.0.4)\n",
      "Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from requests->pooch>=1.0->librosa) (1.25.11)\n",
      "Requirement already satisfied: idna<3,>=2.5 in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from requests->pooch>=1.0->librosa) (2.10)\n",
      "Requirement already satisfied: pycparser in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from cffi>=1.0->soundfile>=0.10.2->librosa) (2.20)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install librosa"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: soundfile in c:\\users\\satvi\\anaconda3\\lib\\site-packages (0.10.3.post1)\n",
      "Requirement already satisfied: cffi>=1.0 in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from soundfile) (1.14.3)\n",
      "Requirement already satisfied: pycparser in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from cffi>=1.0->soundfile) (2.20)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install soundfile"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: numpy in c:\\users\\satvi\\anaconda3\\lib\\site-packages (1.19.2)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install numpy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: sklearn in c:\\users\\satvi\\anaconda3\\lib\\site-packages (0.0)\n",
      "Requirement already satisfied: scikit-learn in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from sklearn) (0.23.2)\n",
      "Requirement already satisfied: numpy>=1.13.3 in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from scikit-learn->sklearn) (1.19.2)\n",
      "Requirement already satisfied: scipy>=0.19.1 in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from scikit-learn->sklearn) (1.5.2)\n",
      "Requirement already satisfied: joblib>=0.11 in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from scikit-learn->sklearn) (0.17.0)\n",
      "Requirement already satisfied: threadpoolctl>=2.0.0 in c:\\users\\satvi\\anaconda3\\lib\\site-packages (from scikit-learn->sklearn) (2.1.0)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install sklearn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting package metadata (current_repodata.json): ...working... done\n",
      "Solving environment: ...working... done\n",
      "\n",
      "# All requested packages already installed.\n",
      "\n",
      "\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "conda install PyAudio"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "import librosa\n",
    "import soundfile\n",
    "import os, glob, pickle\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.neural_network import MLPClassifier\n",
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "def extract_feature(file_name, mfcc, chroma, mel):\n",
    "    with soundfile.SoundFile(file_name) as sound_file:\n",
    "        X = sound_file.read(dtype=\"float32\")\n",
    "        sample_rate=sound_file.samplerate\n",
    "        if chroma:\n",
    "            stft=np.abs(librosa.stft(X))\n",
    "            result=np.array([])\n",
    "        if mfcc:\n",
    "            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)\n",
    "            result=np.hstack((result, mfccs))\n",
    "        if chroma:\n",
    "            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)\n",
    "            result=np.hstack((result, chroma))\n",
    "        if mel:\n",
    "            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)\n",
    "            result=np.hstack((result, mel))\n",
    "        return result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "emotions={\n",
    "  '01':'neutral',\n",
    "  '02':'calm',\n",
    "  '03':'happy',\n",
    "  '04':'sad',\n",
    "  '05':'angry',\n",
    "  '06':'fearful',\n",
    "  '07':'disgust',\n",
    "  '08':'surprised'\n",
    "}\n",
    "observed_emotions=['calm', 'happy', 'fearful', 'disgust']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_data(test_size=0.2):\n",
    "    x,y=[],[]\n",
    "    for file in glob.glob(\"D:\\\\dataset kaggle\\\\Actor_*\\\\*.wav\"):\n",
    "        file_name=os.path.basename(file)\n",
    "        emotion=emotions[file_name.split(\"-\")[2]]\n",
    "        if emotion not in observed_emotions:\n",
    "            continue\n",
    "        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)\n",
    "        x.append(feature)\n",
    "        y.append(emotion)\n",
    "    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train,x_test,y_train,y_test=load_data(test_size=0.25)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(573, 191)\n"
     ]
    }
   ],
   "source": [
    "print((x_train.shape[0], x_test.shape[0]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Features extracted: 180\n"
     ]
    }
   ],
   "source": [
    "print(f'Features extracted: {x_train.shape[1]}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "MLPClassifier(alpha=0.01, batch_size=256, hidden_layer_sizes=(300,),\n",
       "              learning_rate='adaptive', max_iter=500)"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(x_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred=model.predict(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 74.35%\n"
     ]
    }
   ],
   "source": [
    "accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)\n",
    "print(\"Accuracy: {:.2f}%\".format(accuracy*100))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
