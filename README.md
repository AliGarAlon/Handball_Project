# Handball actions classifier:
Access to the dataset: https://ieee-dataport.org/open-access/handball-action-dataset-uniri-hbd
The dataset consists of 751 videos, each containing the performance one of the handball actions out of 7 categories (passing, shooting, jump-shot, dribbling, running, crossing, defence). The videos were manually extracted from longer videos recorded in handball practice sessions. They were recorded in at least full HD (1920 × 1080) resolution at 30 or more frames per second, and mostly one or two players perform the action of interest.
The code for this project consists of 4 different files:
-	EDA_HB.ipynb: this notebook copies the dataset in directory "cleaned_actions" and preprocess the videos in this duplicated folder.
-	deep_CNNmodel_HB.ipynb: this notebook corresponds to first attempt to build a classifier for handball actions. It leverages the EfficientNetB0 to classify the frames of each video, making use of transfer learning. It is the worst performing model.
-	3D_CNNmodel_HB.ipynb: this notebook corresponds to the second attempt of model the classifier. It consists of a Convolutional Neural Network (CNN) built from scratch, that considers not only height x width x colour channels, but also the frames as an additional dimension, thus the “3D”. It performs better than the CNN with transfer learning.
-	CNN-Sequential_model_HB.ipynb: this notebook corresponds to the final attempt for modelling the classifier. It consists of the ResNet101v2 CNN to extract features, followed by a Sequential Neural Network that take into consideration the “time” dimension captured by the different frames. It is the best performing model.
  
# Scaling up of the application:
Storing video (unstructured data) means using a NoSQL database. Due to the flexibility they offer (horizontal scalability, reliability, easy access), using a cloud database like Google Cloud is considered the best option. 
All models have been built with TensorFlow. Hence, when it comes to distributing the training of models, the `tf.distribute.Strategy` API can be used for either data parallelising or model parallelising, according to the needs of the situation. If the case of scaling up the application ever came, Google Kubernetes Engine (GKE) services could be easily used, since Google Cloud is being used as a database. Deploying the application in a Kubernetes cluster would enable to dynamically scale the number of application replicas on demand.

## References:

•	M. Ivasic-Kos and M. Pobar, "Building a labeled dataset for recognition of handball actions using mask R-CNN and STIPS," 2018 7th European Workshop on Visual Information Processing (EUVIP), Tampere, Finland, 2018, pp. 1-6, doi: 10.1109/EUVIP.2018.8611642. 

•	He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016. 

•	Tan, Mingxing, and Quoc Le. "Efficientnet: Rethinking model scaling for convolutional neural networks." International conference on machine learning. PMLR, 2019. 

•	Tran, Du, et al. "A closer look at spatiotemporal convolutions for action recognition." Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. 2018. 

•	Biermann, Henrik, et al. "A unified taxonomy and multimodal dataset for events in invasion games." Proceedings of the 4th International Workshop on Multimedia Content Analysis in Sports. 2021. 

•	He, Kaiming, et al. "Identity mappings in deep residual networks." Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14. Springer International Publishing, 2016. 

•	Montaha, Sidratul, et al. "Timedistributed-cnn-lstm: A hybrid approach combining cnn and lstm to classify brain tumor on 3d mri scans performing ablation study." IEEE Access 10 (2022): 60039-60059. 

•	Chung, Junyoung, et al. "Empirical evaluation of gated recurrent neural networks on sequence modeling." arXiv preprint arXiv:1412.3555 (2014).

## References for code:
https://github.com/AarohiSingla/Video-Classifier-Using-CNN-and-RNN/blob/main/video_classifier_working.ipynb

https://www.tensorflow.org/tutorials/load_data/video

https://www.tensorflow.org/tutorials/video/video_classification

https://www.tensorflow.org/tutorials/video/transfer_learning_with_movinet

https://keras.io/api/applications/resnet/#resnet101v2-function

https://www.analyticsvidhya.com/blog/2022/05/building-a-3d-cnn-in-tensorflow/

https://keras.io/api/layers/recurrent_layers/gru/

https://keras.io/guides/transfer_learning/

https://kubernetes.io/
