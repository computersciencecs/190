anonymous_github now has the problem "The requested file is not found." According to anonymous_github's official Report an issue page, it may be a cache problem. 
Some users pointed out in the post that it can be accessed normally in incognito mode. 
In addition, I found that changing the browser or waiting for a while and then refreshing can access it normally.

workflow:
1-Download the dataset;
2-run split.py to split the dataset;
2-Run top_k_recommendation.py in the LightGCN folder to obtain the generated files needed to build prompt;
3-run graphrag.py to obtain the retrieved KG triples;
4-run make-train-prompt-v.py to obtain the training prompt;
5-run train.py for training;
6-run testprompt.py to obtain the inference prompt;
7-run inference.py for inference;
8-run resultsprocess+evaluation .ipynb to process the inference generated files and calculate metrics.
