# Project Report: Handwritten Text Recognition Using Deep Learning

## 1. Title

Handwritten Text Recognition Using Deep Learning

## 2. Internship Context

This project was developed as part of the CodeAlpha Machine Learning Internship. The objective of the project is to design and implement a deep learning-based Handwritten Text Recognition (HTR) system capable of converting handwritten text images into editable digital text.

The project covers the complete machine learning workflow, including dataset preparation, image preprocessing, model development, training, evaluation, and deployment through a user-friendly Streamlit web application.

The final system uses a CRNN (Convolutional Recurrent Neural Network) architecture consisting of Convolutional Neural Networks (CNNs), Bidirectional Long Short-Term Memory (BiLSTM) networks, and Connectionist Temporal Classification (CTC) decoding for end-to-end handwriting recognition.


## 3. Problem Statement

Handwritten Text Recognition (HTR) is a challenging computer vision and sequence learning problem because handwriting varies significantly between individuals. Variations in writing style, character spacing, slant, stroke thickness, image quality, lighting conditions, and background noise make accurate recognition difficult.

The objective of this project is to develop an automated system that can accurately recognize handwritten words and short sentences from images and convert them into editable digital text.

Example:

```text
Input Image:
Machine Learning is Fun

Predicted Output:
Machine Learning is Fun
```

The proposed solution combines deep learning and image preprocessing techniques to improve recognition performance on real-world handwritten images captured using mobile devices.


## 4. Dataset

The model was trained on a merged handwriting dataset created from three IAM-based sources.

### Datasets Used

1. Local IAM Dataset (Parquet Format)

   * train.parquet
   * validation.parquet
   * test.parquet

2. Bibek130/IAM-line (Hugging Face)

3. alpayariyak/IAM_Sentences (Hugging Face)

The datasets contain handwritten English words, phrases, and sentences paired with their corresponding ground-truth text transcriptions.

### Dataset Processing

During training, the datasets are automatically:

* Loaded from local and Hugging Face sources
* Merged into a unified dataset
* Deduplicated to remove repeated samples
* Cleaned to remove invalid entries
* Used to rebuild a unified character vocabulary

The final dataset contains both word-level and sentence-level handwriting samples, allowing the model to learn character recognition, word formation, and sentence-level context.

The original IAM images are not included in the repository due to dataset licensing restrictions.


## 5. Data Preprocessing

A custom image preprocessing pipeline was developed to improve recognition performance on real-world handwritten photographs.

The preprocessing steps are:

1. Load the input image
2. Convert the image to grayscale
3. Perform shadow and illumination normalization
4. Apply contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
5. Apply adaptive thresholding to separate handwriting from the background
6. Remove small noise components and image artifacts
7. Detect and crop the handwriting region automatically
8. Resize the cropped image while preserving aspect ratio
9. Add padding to create a fixed-size model input
10. Normalize pixel values for neural network processing

### Preprocessing Objectives

The preprocessing pipeline is designed to:

* Reduce lighting variations
* Remove shadows and background noise
* Improve handwriting visibility
* Focus the model on the handwritten region
* Maintain consistent image dimensions for training and inference

The Streamlit application visualizes four preprocessing stages:

1. Grayscale Image
2. Contrast Enhanced Image
3. Thresholded Image
4. Final Processed Image

This preprocessing pipeline significantly improves recognition quality on handwritten photographs captured using mobile devices.


## 6. Model Architecture

The project uses a Convolutional Recurrent Neural Network (CRNN) architecture designed specifically for handwritten text recognition.

The architecture combines convolutional layers for visual feature extraction with recurrent layers for sequence modeling.

### Architecture Components

#### 1. Convolutional Neural Network (CNN)

The CNN extracts visual features from handwritten text images.

Functions:

* Detect character strokes and edges
* Learn local handwriting patterns
* Reduce spatial dimensions while preserving important features

#### 2. Sequence Conversion

The CNN feature map is transformed into a sequence by treating each vertical feature-map column as a time step.

This converts the image recognition problem into a sequence recognition problem.

#### 3. Bidirectional LSTM (BiLSTM)

Bidirectional Long Short-Term Memory layers process the generated sequence in both forward and backward directions.

Advantages:

* Captures left-to-right context
* Captures right-to-left context
* Improves recognition of ambiguous characters

#### 4. Character Classification Layer

A fully connected layer predicts character probabilities for every time step.

The output vocabulary contains all characters observed in the training datasets plus the CTC blank token.

#### 5. CTC Decoder

Connectionist Temporal Classification (CTC) is used to decode character probabilities into readable text without requiring character-level alignment during training.

### Architecture Flow

Input Image

↓

CNN Feature Extraction

↓

Feature Sequence Generation

↓

Bidirectional LSTM

↓

Character Probability Prediction

↓

CTC Decoding

↓

Recognized Text


## 7. Training Strategy

The model was trained using PyTorch and optimized for handwritten text recognition using the CTC learning framework.

### Training Configuration

* Framework: PyTorch
* Optimizer: AdamW
* Loss Function: CTC Loss
* Device Support: CPU and GPU
* Batch Processing: Dynamic batch loading using DataLoader
* Checkpoint Saving: Best model automatically saved

### Training Process

1. Load merged handwriting datasets
2. Apply preprocessing and vocabulary encoding
3. Generate batches using PyTorch DataLoader
4. Train the CRNN model using CTC Loss
5. Evaluate performance on the validation dataset
6. Compute CER and WER metrics
7. Save the best-performing checkpoint
8. Reduce learning rate when validation performance plateaus

### Learning Rate Scheduling

The project uses ReduceLROnPlateau scheduling to automatically reduce the learning rate when validation Character Error Rate (CER) stops improving.

### Model Checkpoint

The best model is stored as:

```text
saved_models/best_crnn_ctc.pt
```

This checkpoint is later used by the prediction pipeline and Streamlit application.


## 8. Evaluation Metrics

The model performance is evaluated using multiple handwriting-recognition metrics.

### 1. Character Error Rate (CER)

CER measures the percentage of character-level mistakes.

It is calculated using edit distance between the predicted text and ground-truth text.

Lower CER indicates better recognition performance.

### 2. Word Error Rate (WER)

WER measures recognition accuracy at the word level.

It counts substitutions, insertions, and deletions of words.

Lower WER indicates better sentence-level recognition.

### 3. Exact Match Accuracy

Exact Match Accuracy measures the percentage of samples where the entire prediction exactly matches the ground-truth transcription.

This metric is stricter than CER and WER because even a single incorrect character causes the prediction to be considered incorrect.

### 4. Validation Loss

CTC Loss is monitored throughout training to measure overall model learning progress.

### Purpose of Using Multiple Metrics

* CER evaluates character-level performance.
* WER evaluates word-level performance.
* Exact Match Accuracy evaluates complete transcription correctness.
* Validation Loss monitors training stability.

Together these metrics provide a comprehensive assessment of handwriting recognition quality.


## 9. Prediction Pipeline

The prediction system converts a handwritten image into editable text using the trained CRNN model.

### Prediction Workflow

1. User uploads a handwritten image.
2. The image is converted to grayscale.
3. Shadow and illumination normalization are applied.
4. Contrast enhancement is performed using CLAHE.
5. Adaptive thresholding separates handwriting from the background.
6. Noise removal and handwriting-region cropping are performed.
7. The image is resized and padded to match model input dimensions.
8. The processed image is passed through the trained CRNN model.
9. Character probabilities are generated.
10. Greedy CTC decoding converts probabilities into text.
11. The recognized text and confidence score are displayed.

### Supported Input

The system works best with:

* Plain white paper
* Dark handwritten text
* Medium to large handwriting
* Well-lit images
* Handwritten words and short sentences

### Output

The prediction module returns:

* Recognized text
* Confidence score
* Intermediate preprocessing stages
* Downloadable text output


## 10. Streamlit Web Application

A Streamlit-based web application was developed to provide an interactive interface for handwritten text recognition.

### Features

* Upload handwritten text images
* Display original uploaded image
* Display preprocessing stages
* Display recognized text
* Show prediction confidence score
* Download recognized text as a TXT file
* Enable or disable photo-cleanup preprocessing

### Preprocessing Visualization

The application displays four preprocessing stages:

1. Grayscale Image
2. Contrast Enhanced Image
3. Thresholded Image
4. Final Processed Image

This allows users to understand how the image is transformed before recognition.

### User Guidelines Displayed in the Application

The interface provides recommendations for achieving the best recognition results:

* Use plain white paper
* Avoid ruled or lined notebook pages
* Write clearly using dark ink
* Keep handwriting reasonably large
* Use good lighting conditions
* Prefer handwritten words or short sentences

### Benefits

The Streamlit application enables real-time testing of the trained model without requiring users to run Python scripts manually.


## 11. Results

The trained CRNN model was evaluated using validation metrics and real-world handwritten image testing through the Streamlit application.

### Final Validation Results

| Metric                          | Value  |
| ------------------------------- | ------ |
| Validation Loss                 | 0.2505 |
| Validation CER                  | 0.0610 |
| Validation WER                  | 0.2184 |
| Validation Exact Match Accuracy | 16.29% |

### Training Summary

* Total Training Epochs: 50
* Best Model Saved Automatically
* Vocabulary Size: 80 Characters
* Trainable Parameters: Approximately 5.8 Million

### Real-World Prediction Results

The model was tested on handwritten images captured using a mobile phone and uploaded through the Streamlit application.

Observed performance:

* Successfully recognized most handwritten words and short sentences.
* Achieved confidence scores generally above 90% on clean handwritten images tested through the Streamlit application.
* Worked effectively on different handwriting styles.
* Maintained good performance after preprocessing and automatic handwriting-region detection.

### Representative Prediction Outcomes

| Input Text                               | Predicted Output                         |
| ---------------------------------------- | ---------------------------------------- |
| hey i am sam                             | hey t am sam                             |
| The cat is sleeping on the sofa.         | The cat is sleeping on the sofo.         |
| She enjoys reading books in the evening. | She engays reading books in the eveniong |

The following examples illustrate typical model behavior on real handwritten images. Minor character-level errors may still occur in challenging handwriting styles.

The results demonstrate that the model can accurately recognize handwritten text while maintaining low character-level error rates.

### Discussion

The model performs best when:

* Handwriting is written on plain white paper.
* Text is clearly visible and reasonably large.
* Images are well lit and contain minimal shadows.

Performance decreases when:

* Handwriting is highly cursive or decorative.
* Images contain strong shadows or blur.
* Very long sentences are compressed into a small image width.

### Streamlit Application Interface

![Streamlit Interface](screenshots/interface.png)

Figure 1: Main interface of the handwritten text recognition application.

### Prediction Example 1

![Prediction Example 1](screenshots/example1.png)

Figure 2: Successful recognition of a handwritten phrase.

### Prediction Example 2

![Prediction Example 2](screenshots/example2.png)

Figure 3: Recognition result for a short handwritten sentence.

### Prediction Example 3

![Prediction Example 3](screenshots/example3.png)

Figure 4: Recognition result for a cursive handwriting sample.



## 12. Error Analysis

Although the model achieves strong recognition performance on clean handwriting images, several common error patterns were observed during testing.

### Character-Level Errors

Characters with similar visual appearances may occasionally be confused.

Examples:

* l ↔ I
* o ↔ a
* m ↔ rn
* c ↔ e

These errors are common in handwriting recognition systems because handwritten character shapes vary significantly across writers.

### Word-Level Errors

Some predictions contain minor spelling deviations despite preserving the overall meaning.

Examples observed during testing:

* sofa → sofo
* enjoys → engays

These mistakes typically occur when characters are partially merged, faint, or unusually shaped.

### Image Quality Issues

Recognition performance decreases when:

* Images contain shadows.
* Handwriting is very small.
* The background contains notebook lines.
* Images are blurry or compressed.

### Sentence-Length Limitations

The model was primarily trained for words and short sentences.

Very long text lines may experience:

* Character omissions
* Missing spaces
* Reduced confidence scores

### Future Improvements

Potential improvements include:

* Larger and more diverse handwriting datasets
* Additional data augmentation
* Beam-search CTC decoding
* Language-model-assisted correction
* Higher-resolution sentence-level training
* GPU training for additional epochs

Despite these limitations, the model demonstrates reliable performance for handwritten words and short sentences captured using mobile devices.


## 13. Conclusion

This project successfully developed a complete Handwritten Text Recognition system using deep learning techniques.

A CRNN architecture combining CNN feature extraction, Bidirectional LSTM sequence modeling, and CTC decoding was implemented and trained on a merged IAM-based handwriting dataset. A custom preprocessing pipeline was developed to improve recognition quality on real-world handwritten photographs.

The trained model achieved low Character Error Rate (CER) and demonstrated strong performance on handwritten words and short sentences. The system was further deployed through a Streamlit web application that allows users to upload handwritten images and obtain editable digital text in real time.

The project demonstrates the practical application of deep learning, computer vision, sequence modeling, and OCR technologies for handwriting recognition tasks.


## 14. GitHub Summary

Developed a Handwritten Text Recognition (HTR) system using PyTorch, OpenCV, and Streamlit. Implemented a CRNN architecture combining CNN feature extraction, Bidirectional LSTM sequence modeling, and CTC decoding for end-to-end handwriting recognition. Built a custom image preprocessing pipeline including shadow removal, contrast enhancement, adaptive thresholding, noise removal, and handwriting-region detection. Trained and evaluated the model on merged IAM-based handwriting datasets using Character Error Rate (CER) and Word Error Rate (WER). Deployed the solution through an interactive Streamlit web application capable of converting handwritten images into editable digital text.


