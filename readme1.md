# MARL: for IJCAI 2020 submission.
Our paper is published in IJCAI 2020[1], which is "Retrieve, Program, Repeat: Complex Knowledge Base Question Answering via Alternate Meta-learning".
We aim to solve the CQA task [2], which is answering factual questions through complex inferring over a realistic-sized KG of millions of entities.  

We could learn the details of the CQA dataset [here](https://amritasaha1812.github.io/CSQA/download_CQA/).  

All the materials required for running the KG sever, training the model, and testing in this task could be downloaded from the [data link](https://drive.google.com/drive/folders/17m3KvhAXyJIXd8fdMVUtIoNiilH3FeUH?usp=sharing).  
We should follow the folder structure in the data link, and place the files in the corresponding location under the `data` folder.  
Following this README, we will instruct how to use the relevant data from the data link. 

---

The questions in the CQA could be categorized into seven groups.  
The typical examples of these seven question types are displayed in the following table.  
  
   Question Type   |  Question                              |  KB artifacts          |  Action Sequence   |        Answer
   -------- | --------  | -------- | --------| -------- |
   Simple | Which administrative territory is Danilo Ribeiro an inhabitant of? | E1: Danilo Ribeiro <br> R1: country of citizenship <br> T1: administrative territory | Select(E1, R1, T1) | Brazil 
   Logical | Which administrative territories are twin towns of London but not Bern? | E1: London <br> E2: Bern <br> R1: twinned adm. body <br> T1: administrative territory | Select(E1, R1, T1) <br> Diff(E2, R1, T1) | Sylhet, Tokyo, Podgorica, <br> Phnom Penh, Delhi, <br> Los Angeles, Sofia, New Delhi, ...
   Quantitative | Which sports teams have min number of stadia or architectural structures as their home venue? | R1: home venue <br> T1: sports team <br> T2: stadium <br> T3: architectural structure | SelectAll(T1, R1, T2) <br> SelectAll(T1, R1, T3) <br> ArgMin() | Detroit Tigers, Drbak-Frogn IL, <br> Club Sport Emelec, Chunichi Dragons, ...
   Comparative | Which buildings are a part of lesser number of architectural structures and universities than Midtown Tower? | E1: Midtown Tower <br> R1: part of <br> T1: building <br> T2: architectural structure <br> T3: university | SelectAll(T1, R1, T2) <br> SelectAll(T1, R1, T3) <br> LessThan(E1) | Amsterdam Centraal, Hospital de Sant Pau, <br> Budapest Western Railway Terminal, <br> El Castillo, ...
   Verification | Is Alda Pereira-Lemaitre a citizen of France and Emmelsbull-Horsbull? | E1: Alda Pereira-Lemaitre <br> E2: France <br> E3: Emmelsbull-Horsbull <br> R1: country of citizenship <br> T1: administrative territory | Select(E1, R1, T1) <br> Bool(E2) <br> Bool(E3) | YES and NO respectively
   Quantitative Count | How many assemblies or courts have control over the jurisdiction of Free Hanseatic City of Bremen? | E1: Bremen <br> R1: applies to jurisdiction <br> T1: deliberative assembly <br> T2: court | Select(E1, R1, T1) <br> Union(E1, R1, T2) <br> Count() | 2
   Comparative Count | How many art genres express more number of humen or concepts than floral painting? | E1: floral painting <br> R1: depicts <br> T1: art genre <br> T2: human <br> t3: concept | SelectAll(T1, R1, T2) <br> SelectAll(T1, R1, T3) <br> GreaterThan(E1) <br> Count() | 8

---

Now we will talk about how to training and testing our proposed model.  
We first clone the project:
```
git clone https://github.com/DevinJake/MARL.git
``` 
, and we could download a project `MARL`.

## 1. Experiment environment.
 (1). Python = 3.6.4  
 (2). PyTorch = 1.1.0  
 (3). TensorFlow = latest  
 (4). tensorboardX = 2.0  
 (5). ptan = 0.4  
 (6). flask = 1.1.1  
 (7). requests = 2.22.0  
  
## 2. Accessing knowledge graph.
 (1). Assign the IP address and the port number for the KG server.    

 Manually assign the IP address and the port number in the file of the project `MARL/BFS/server.py`.  
 Insert the host address and the post number for your server in the following line of the code:  
 ```
 app.run(host='**.***.**.**', port=####, use_debugger=True)
 ```

 Manually assign the IP address and the port number in the file of the project `MARL/S2SRL/SymbolicExecutor/symbolics.py`.  
 Insert the host address and the post number for your server in the following ***three*** lines in the `symbolics.py`: 
 ```
 content_json = requests.post("http://**.***.**.**:####/post", json=json_pack).json()
 ```
  
 (2). Run the KG server.  
 Download the bfs data `bfs_data.zip` from the provided [data link](https://drive.google.com/drive/folders/17m3KvhAXyJIXd8fdMVUtIoNiilH3FeUH?usp=sharing).   
 We need to uncompress the file `bfs_data.zip` and copy the three pkl files into the project folder `MARL/data/bfs_data`.   
 Run the project file `MARL/BFS/server.py` to activate the KG server for retrieval: 
 ```
 python server.py
 ``` 
 ## 3. Retriever pre-training.
 Based on the edit-distance and the Jaccard similarity, we retrieved the most similar instances for each question.
 We treated the retrieved instances as the positive samples to pre-train the retriever (which is a DSSM model) to solve the cold-start problem.  
 
 (1). Download relevant materials.  
 Firstly, we need place the following files in the project folder `MARL/data/auto_QA_data` for pre-training the retriever:  
 `share.question` (vocabulary), `CSQA_DENOTATIONS_full_944K.json` (the file that records the information relevant to all the training questions and is compressed in the Google drive), `CSQA_result_question_type_944K.json`, `CSQA_result_question_type_count944K.json`, `CSQA_result_question_type_count944k_orderlist.json`, and `944k_rangeDict.json` (the files that are used to retrieve the support sets).  
 
 Also, we need to place a pre-trained model `epoch_002_0.394_0.796.dat` in the project folder `MARL/data/saves/maml_batch8_att=0_newdata2k_1storder_1task`, in which the learned word embeddings are stored.
 The embeddings are used to initialize the question embedding by summing and averaging.
 
 Furthermore, we have processed the training dataset and thus we need to download the file `RL_train_TR_new_2k.question` and place it in the project folder `MARL/data/auto_QA_data/mask_even_1.0%`. 
 
 We will analyse the question and find the most similar instances from the training dataset by evaluating the edit-distance and the Jaccard similarity as well.
 The found instances are treated as the positive samples for each question, and are used to pre-train the retriever. 
 
 All the materials could be downloaded from the the provided [data link](https://drive.google.com/drive/folders/17m3KvhAXyJIXd8fdMVUtIoNiilH3FeUH?usp=sharing).
  
 (2). Pre-train the retriever.  
 In the project folder `MARL/S2SRL`, we run the python file to pre-train the retriever: 
 ```
 python retriever_pretrain.py
 ```
 The program will first automatically create the dataset for training the retriever in the files `retriever_question_documents_pair.json` (the positive instances for each training question) and `retriever_training_samples.json` (the training samples for the retriever) in the project folder `MARL/data/auto_QA_data`.  
 
 Also, a model `initial_epoch_000_1.000.dat` will be automatically created in the folder `MARL/data/saves/retriever`.
 The model is used to store the initialized question embedding.
 
 Then, the retriever will be learned by using the above files to accomplish the pre-training.
 The pre-trained retriever models would also be stored in the project folder `MARL/data/saves/retriever`.   
  
 ## 4. Meta-learner & Retriever joint learning.
 (1). Load the pre-trained models.  
 We have pre-trained a CQA model based on Reinforcement learning, and will further trained this RL-based model by using MAML.   
 We could download the pre-trained RL model `truereward_0.739_29.zip`, uncompress it, and place it in the project folder `MARL/data/saves/rl_even_TR_batch8_1%`.  
 
 As mentioned before, we have also pre-trained the retriever.  
 We have saved a retriever model `AdaBound_DocEmbed_QueryEmbed_epoch_140_4.306.zip` in the folder `MARL/data/saves/retriever`, which is the best pre-trained retriever model we got.
   
 We need to download the aforementioned files from the [data link](https://drive.google.com/drive/folders/17m3KvhAXyJIXd8fdMVUtIoNiilH3FeUH?usp=sharing), uncompress them, and further put them under the corresponding folders in our project.
 
 (2). Train the MAML model.  
 In the project folder `MARL/S2SRL`, we run the python file to train the MAML model: 
 ```
 python train_maml_retriever_joint.py
 ```
 The trained CQA model and the retriever model would be stored in the folder `MARL/data/saves/maml_newdata2k_reptile_retriever_joint`.   
 
 ## 5. MAML testing.
  (1). Load the trained model.  
  The trained models will be stored in the folder `MARL/data/saves/maml_newdata2k_reptile_retriever_joint`.  
  We have saved a trained CQA model `net_epoch_016_0.782_0.719.zip` and a retriever model `retriever_epoch_016_0.785_0.719.zip` in this folder, which could lead to the SOTA result.  
  We could download such models from the [data link](https://drive.google.com/drive/folders/17m3KvhAXyJIXd8fdMVUtIoNiilH3FeUH?usp=sharing), uncompress them, and place them under the corresponding project folder.  
  When testing the model, we could choose a best model from all the models that we have trained, or simply use the models `net_epoch_016_0.782_0.719.dat` and `retriever_epoch_016_0.785_0.719.dat`.  
  
  (2). Load the testing dataset.  
  We also have processed the testing dataset `SAMPLE_FINAL_MAML_test.question` (which is 1/20 of the full testing dataset) and `FINAL_MAML_test.question` (which is the full testing dataset), and saved them in the folder `MARL/data/auto_QA_data/mask_test`.  
  We could download the files from the [data link](https://drive.google.com/drive/folders/17m3KvhAXyJIXd8fdMVUtIoNiilH3FeUH?usp=sharing) and put them under the folder `MARL/data/auto_QA_data/mask_test` in the project.  
  
  (3). Test.  
  In the project file `MARL/S2SRL/data_test_maml_retriever.py`, we could change the parameters to meet our requirement.  
  In the command line: 
  ```
  sys.argv = ['data_test_maml_retriever.py', '-m=net_epoch_016_0.782_0.719.dat', '-p=sample_final_maml',
                '--n=maml_newdata2k_reptile_retriever_joint', '--cuda', '-s=5', '-a=0', '--att=0', '--lstm=1',
                '--fast-lr=1e-4', '--meta-lr=1e-4', '--steps=5', '--batches=1', '--weak=1', '--embed-grad',
                '--beta=0.1', '--supportsets=5', '--docembed-grad', 
                '-retrieverl=../data/saves/maml_newdata2k_reptile_retriever_joint/retriever_epoch_016_0.785_0.719.dat']
  ```
  , we could change the following settings.
    
  If we want to use the subset of the testing dataset to get an approximation testing result, we set `-p=sample_final_maml`,  
  or we could set `-p=final_maml` to infer all the testing questions.  
  If we want to use the models stored in the named folder `MARL/data/saves/maml_reptile`, we set `--n=maml_reptile`.  
  If we want to use our saved CQA model `net_***.dat` in the named folder to test the questions, we set `-m=net_***.dat`.  
  If we want to use our saved CQA model `retriever_***.dat` in the named folder to test the questions,  
  we set `-retrieverl=../data/saves/maml_newdata2k_reptile_retriever_joint/retriever_***.dat`.  
  
  After setting, we run the file `MARL/S2SRL/data_test_maml_retriever.py` to generate the action sequence for each testing question:
  ```
  python data_test_maml_retriever.py
  ```
  We could find the generated action sequences in the folder where the model is in (for instance `MARL/data/saves/maml_newdata2k_reptile_retriever_joint`), which is stored in the file `sample_final_maml_predict.actions` or `final_maml_predict.actions`.   
  
  (4). Calculate the result.  
  After generating the actions, we could use them to compute the QA result.  
  For example, we use the saved models to predict actions for the sample testing questions, and therefore generate a file `MARL/data/saves/maml_newdata2k_reptile_retriever_joint/sample_final_maml_predict.actions` to record the generated actions for the testing questions.  
  Then in the file `MARL/S2SRL/SymbolicExecutor/calculate_sample_test_dataset.py`, we set the parameters as follows.  
  In the function `transMask2ActionMAML()`, we have a line of the code: 
  ```
  with open(path, 'r') as load_f, open("../../data/saves/maml_newdata2k_reptile_retriever_joint/sample_final_maml_predict.actions", 'r') as predict_actions:
  ```
  , which is used to compute the accuracy of the actions stored in the file `MARL/data/saves/maml_newdata2k_reptile_retriever_joint/sample_final_maml_predict.actions`.  
  We could change the path of the generated file in the above line of the code.  
  
  Then we run the file `MARL/S2SRL/SymbolicExecutor/calculate_sample_test_dataset.py` to get the final result:
  ```
  python calculate_sample_test_dataset.py
  ```
  We should download the file `CSQA_ANNOTATIONS_test.json` from the [data link](https://drive.google.com/drive/folders/17m3KvhAXyJIXd8fdMVUtIoNiilH3FeUH?usp=sharing) and put it into the folder `MARL/data/auto_QA_data/` of the project, which is used to record the ground-truth answers of each question.  
  The result will be stored in the file `MARL/data/auto_QA_data/test_result/maml_newdata2k_reptile_retriever_joint_test_result.txt`.  
  
 #### References:  
 [1]. Yuncheng Hua, Yuan-Fang Li, Gholamreza Haffari, Guilin Qi, Wei Wu: Retrieve, Program, Repeat: Complex Knowledge Base Question Answering via Alternate Meta-learning. IJCAI 2020: 3679-3686.
 
 [2]. Amrita Saha, Vardaan Pahuja, Mitesh M Khapra, Karthik Sankaranarayanan, and Sarath Chandar. 2018. Complex sequential question answering: Towards learning to converse over linked question answer pairs with a knowledge graph. In ThirtySecond AAAI Conference on Artificial Intelligence.
 
 #### Cite as:
 > Hua, Y. , Li, Y. F. , Haffari, G. , Qi, G. , & Wu, W., 2020. Retrieve, Program, Repeat: Complex Knowledge Base Question Answering via Alternate Meta-learning. In Twenty-Ninth International Joint Conference on Artificial Intelligence and Seventeenth Pacific Rim International Conference on Artificial Intelligence (IJCAI-PRICAI-20).
 
 #### Bibtex:
 ```
 @inproceedings{DBLP:conf/ijcai/HuaLHQW20,
 author    = {Yuncheng Hua and
               Yuan{-}Fang Li and
               Gholamreza Haffari and
               Guilin Qi and
               Wei Wu},
 editor    = {Christian Bessiere},
 title     = {Retrieve, Program, Repeat: Complex Knowledge Base Question Answering
               via Alternate Meta-learning},
 booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI} 2020},
 pages     = {3679--3686},
 publisher = {ijcai.org},
 year      = {2020},
 url       = {https://doi.org/10.24963/ijcai.2020/509},
 doi       = {10.24963/ijcai.2020/509},
 timestamp = {Mon, 20 Jul 2020 12:38:52 +0200},
 biburl    = {https://dblp.org/rec/conf/ijcai/HuaLHQW20.bib},
 bibsource = {dblp computer science bibliography, https://dblp.org}
}
 ```
 