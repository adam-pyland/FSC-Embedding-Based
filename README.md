1. Git clone this repo
2. Either install requirements.txt or use docker image ultralytics:latest (prefered)
3. Create folder ultralytics in the repo and unzip ultralytics.zip in it
4. Config/change in ultralytics folder, run_yv11_pred.py to create crops (single object images)
5. In home folder (Sultan) use extract_f_vectors.py to produce feature vectors
6. If one wants to compare feature vectors: use compare_F_vectors.py using cosine similarity
7. For clustering I used tsne (tsne_f_vector.py)
8. To train one class SVM (with your common class) and predict  with rare or mixture of rare/common class, use train_OC-SVM_with_pred.py. 
