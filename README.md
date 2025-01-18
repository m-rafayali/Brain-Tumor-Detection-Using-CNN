# **Brain Tumor Detection Using CNN**

![Project Banner]([https://via.placeholder.com/1200x400.png?text=Brain+Tumor+Detection+Using+CNN](https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41598-020-74419-9/MediaObjects/41598_2020_74419_Fig2_HTML.jpg))

## **Overview**
Brain tumors are one of the most critical medical conditions requiring accurate diagnosis for effective treatment. This project leverages the power of **Convolutional Neural Networks (CNN)** to detect brain tumors from MRI images. The objective is to automate and enhance the detection process, aiding healthcare professionals with quick and reliable results.

---

## **Motivation**
Detecting brain tumors early can save lives. Manual analysis of MRI scans is time-consuming and prone to human error. This project uses **Artificial Intelligence (AI)** and **Deep Learning** techniques to:
- Improve accuracy in tumor detection.
- Reduce diagnosis time.
- Aid medical professionals in making informed decisions.

---

## **Key Features**
- **Automated Tumor Detection**: Classifies MRI images as tumor or non-tumor.
- **High Accuracy**: Built with advanced CNN architectures.
- **User-Friendly Interface**: Simple, clean, and efficient.
- **Transfer Learning**: Utilizes pre-trained models to enhance performance.

---

## **Approach**
1. **Data Collection**: MRI images sourced from publicly available datasets.
2. **Preprocessing**: Images normalized, resized, and augmented for robust model training.
3. **Model Building**: A CNN architecture was designed and trained using TensorFlow/Keras.
4. **Evaluation**: The model was evaluated on unseen test data for accuracy and performance metrics.
5. **Deployment**: The trained model can be deployed in real-world applications.

---

## **Dataset**
The dataset consists of MRI images classified into:
- **Tumor**: Images containing brain tumors.
- **Non-Tumor**: Images without brain tumors.

**Source**: [Brain MRI Images Dataset (Kaggle)](https://www.kaggle.com)

| **Category** | **No. of Images** |
|--------------|--------------------|
| Tumor        | 1550              |
| Non-Tumor    | 1550              |
| **Total**    | **3100**          |

---

## **Technologies Used**
- **Programming Language**: Python
- **Libraries/Frameworks**:
  - TensorFlow/Keras
  - OpenCV
  - NumPy
  - Matplotlib
- **Hardware**:
  - Google Colab GPU for model training
- **Version Control**: Git & GitHub

---

## **Model Architecture**
- **Convolutional Layers**: Extract spatial features from the MRI images.
- **Pooling Layers**: Down-sample feature maps to reduce complexity.
- **Fully Connected Layers**: Classify the input as tumor or non-tumor.
- **Activation Functions**: `ReLU`, `Softmax`
- **Loss Function**: `Categorical Crossentropy`
- **Optimizer**: `Adam`

---

## **Results**
| **Metric**           | **Value**   |
|-----------------------|-------------|
| **Accuracy**          | 96.2%      |
| **Precision**         | 94.8%      |
| **Recall**            | 95.5%      |
| **F1-Score**          | 95.1%      |

---

## **Usage**
### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/m-rafayali/Brain-Tumor-Detection-Using-CNN.git
   cd Brain-Tumor-Detection-Using-CNN

## **Project Structure**
Brain-Tumor-Detection-Using-CNN/
├── data/                 # Dataset folder
├── model/                # Trained model files
├── notebooks/            # Jupyter notebooks for experiments
├── src/                  # Source code
├── main.py               # Main script for running the model
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

## **Future Enhancement**
- Support for multi-class classification (e.g., glioma, meningioma).
- Integration with mobile applications for real-time detection.
- Improved preprocessing for noisy datasets.
- Deployment on cloud platforms for scalability.

## **Contributing**
Contributions are welcome! Please fork the repository and submit a pull request. For major changes, open an issue first to discuss your ideas.

##**License**
This project is licensed under the MIT License. See the LICENSE file for details.

## **Contact**
For any inquiries, reach out to:
Muhammad Rafay Ali
   - Email: m.rafayali@outlook.com
   - GitHub: @m-rafayali

## **Acknowledgments**
   - Dataset provided by Kaggle.
   - Thanks to TensorFlow and Keras communities for their tools and documentation.
   - Special thanks to all contributors and collaborators.
    
**Disclaimer:** This project is for educational purposes only and is not intended for clinical use.
