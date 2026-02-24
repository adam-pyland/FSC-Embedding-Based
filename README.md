# Ariel's Code

1. Git clone this repo
2. Either install requirements.txt or use docker image ultralytics:latest (prefered)
3. Create folder ultralytics in the repo and unzip ultralytics.zip in it
4. Config/change in ultralytics folder, run_yv11_pred.py to create crops (single object images)
5. In home folder (Sultan) use extract_f_vectors.py to produce feature vectors
6. If one wants to compare feature vectors: use compare_F_vectors.py using cosine similarity
7. For clustering I used tsne (tsne_f_vector.py)
8. To train one class SVM (with your common class) and predict  with rare or mixture of rare/common class, use train_OC-SVM_with_pred.py. 


# Adam's Code:
```text
Check_Separability_of_Novel_Base_Classes.py
    - Trains logistinc Regression from the embeddings feature vectors.
Detecton_with_SAHI.py
    - Detect Objects with Tiles
Detector.py
    - Detection, Manual Tiling.
Detector_Evaluation.py
    - Original Code Detection.
Embedding.py
    - Embed the Crops to feature vectors Using DINOv3.
Feature_Adapter.py
    - Train a small Feature Adapter Network to learn to seaprate the features better.
    - Works on the training data,
Feature_Adapter_CHECK_IF_WORKING.py
    - Check if the feature adapter works on the Validation dataset only.
T-SNE-visualize-features.py
    - Visualize the feature vectors seapration.
```

# Folders
```text
├───Adapted_Features
    - Output of the Feature Adapter code.

├───DINO_FAIR1M_features
    - Features Embedded with DINOv3.

├───Input_Features
    - Features of the crops embedded with Resnet50.

├───Original_Base_Class_Img_Crops
    - Crops of the Base Class objects using the Ground Truth Annotations

├───Original_Novel_Class_Img_Crops
    - Crops of the Novel Class objects using the Ground Truth Annotations

├───preprocess_codes
    - Preprocess codes of the input dataset

├───Vehicles_Base_Class
    - Full images of Validation.

└───Vehicles_Novel_Class
    - Full images of Validation.
```