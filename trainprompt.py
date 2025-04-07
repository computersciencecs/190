import sys
import random
import pandas as pd
import jsonlines
import numpy as np
import copy
import pickle
random.seed(42)
np.random.seed(42)

dataset_names = ['ml-1m']
isHint = True
sample_method = 'uniform'  

output_df = pd.read_csv('output22.csv', sep='|', header=None, names=['movie_id', 'movie_name', 'encoding'])
output_movie_id_types = output_df['movie_id'].apply(type).unique()
entity_df = pd.read_csv('only_entity-id.tsv', sep='\t', header=None, names=['encoding', 'translate1', 'translate2', 'translate3', 'entity_id'])
kg_less_test_df = pd.read_csv('kg_less_test_filtered.tsv', sep='\t', header=None, names=['head', 'relation', 'tail'])
kg_less_test_id_df = pd.read_csv('kg_less_test_id_filtered.tsv', sep='\t', header=None, names=['head_id', 'relation_id', 'tail_id'])
kg_head_id_types = kg_less_test_id_df['head_id'].apply(type).unique()
kg_tail_id_types = kg_less_test_id_df['tail_id'].apply(type).unique()
relations_df = pd.read_csv('all_relations_id.tsv', sep='\t', header=None, names=['relation_id', 'relation'])
movie_name_dict = dict(zip(output_df['movie_id'], output_df['movie_name']))
movie_id_to_encoding = dict(zip(output_df['movie_id'], output_df['encoding']))
entity_id_to_name = dict(zip(entity_df['entity_id'], entity_df['translate1']))
relation_id_to_description = dict(zip(relations_df['relation_id'], relations_df['relation']))

def get_kg_triples_like_1(book_list, kg_id_df, kg_text_df, entity_to_name, relation_to_desc):
    triples = []
    
    for book in book_list:
        related_triples = kg_id_df[(kg_id_df['head_id'] == book) | (kg_id_df['tail_id'] == book)]
        
        if related_triples.empty:
            print(f"no for book_id {book}")
        else:
            selected_triples = related_triples.sample(n=min(1, len(related_triples)), random_state=42)
            
            for _, triple in selected_triples.iterrows():
                head_text = kg_text_df.iloc[triple.name]['head']
                relation_text = kg_text_df.iloc[triple.name]['relation']
                tail_text = kg_text_df.iloc[triple.name]['tail']
                triples.append(f'{head_text} {relation_text} {tail_text}')
                #triples.append(f'{head_text} - {relation_text} - {tail_text}')
    
    return triples


def get_kg_triples_wait_1(book_list, kg_id_df, kg_text_df, entity_to_name, relation_to_desc):
    triples = []
    
    for book in book_list:
        related_triples = kg_id_df[(kg_id_df['head_id'] == book) | (kg_id_df['tail_id'] == book)]
        
        if related_triples.empty:
            print(f"未找到相关三元组 for book_id {book}")
        else:
            selected_triples = related_triples.sample(n=min(1, len(related_triples)), random_state=42)
            
            for _, triple in selected_triples.iterrows():
                head_text = kg_text_df.iloc[triple.name]['head']
                relation_text = kg_text_df.iloc[triple.name]['relation']
                tail_text = kg_text_df.iloc[triple.name]['tail']
                triples.append(f'{head_text} {relation_text} {tail_text}')
                #triples.append(f'{head_text} - {relation_text} - {tail_text}')
    
    return triples
    
def sort_list_reverse_with_indices(lst):
    sorted_indices = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)
    sorted_indices = [index for index, _ in sorted_indices]
    return sorted_indices




for dataset_name in dataset_names:

    with open('./ml-1m_item.pkl', "rb") as file:
        cm_item = pickle.load(file)
    with open('./ml-1m_user.pkl', "rb") as file:
        cm_user = pickle.load(file)
    with open('./ml-1m_pred.pkl', "rb") as file:
        cm_pred = pickle.load(file)
    
    with open('./ml-1m_item_id_mapping.pkl', "rb") as file:
        mf_item = pickle.load(file)
    with open('./ml-1m_user_id_mapping.pkl', "rb") as file:
        mf_user = pickle.load(file)
    with open('./ml-1m_rating_matrix.pkl', "rb") as file:
        mf_pred = pickle.load(file)

    with open('./ml-1m_user_emb.pkl', "rb") as file:
        cm_user_emb = pickle.load(file)

    kws = 'movie'

    if dataset_name == 'ml-1m':
        df_like = pd.read_csv('./train_set.txt', names=['u', 'i', 'r', 't'], sep=' ')
        df_dislike = pd.read_csv('./dislike.txt', header=None, names=['u', 'i', 'r', 't'])
        movie_info = pd.read_csv('./movie_info.csv', header=None,names=['movie_id', 'movie_name', 'year', 'genre'], sep='|', engine='python', encoding='latin-1')
        df_like_p = pd.read_csv('./train_set.txt', sep=' ')
        df_like_p.columns = ['u', 'i', 'r', 't']

        movie_id_list = movie_info['movie_id'].tolist()
        movie_name_list = movie_info['movie_name'].tolist()
        movie_name_dict.update({movie_id_list[i]: movie_name_list[i] for i in range(len(movie_name_list))})

    def sort_list_reverse_with_indices(lst):
        sorted_indices = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)
        sorted_indices = [index for index, _ in sorted_indices]
        return sorted_indices

    mes_list = []
    gt_list = []


    if 'ml-1M' in dataset_name:
        sample_n = 1000
    else:
        sample_n = 1000
    user_list = list(df_like['u'].unique())
    sample_list = []
    import math

    weights = [math.log(len(df_like[df_like['u'] == uni])) for uni in user_list]

    if sample_method == 'uniform':
        for i in range(sample_n):
            sample_ = random.sample(user_list, 1)[0]
            sample_list.append(sample_)
    else:
        sample_list1 = []
        sample_list2 = []

        sample_imp = int(sample_n * 0.6)
        for i in range(sample_imp):
            sample_ = random.choices(user_list, weights, k=1)[0]
            sample_list1.append(sample_)

        from sklearn.cluster import KMeans


        kmeans = KMeans(n_clusters=10, random_state=0).fit(cm_user_emb)
        labels = kmeans.labels_

        counts = np.bincount(labels)

        samples_per_cluster = np.round(counts / counts.sum() * sample_imp).astype(int)

        sampled_ids = []
        for cluster_id, samples in enumerate(samples_per_cluster):
            cluster_ids = np.where(labels == cluster_id)[0]
            sampled_ids.extend(np.random.choice(cluster_ids, samples, replace=True))

        mf_user_i = {value: key for key, value in mf_user.items()}
        sampled_ids = [mf_user_i[_] for _ in sampled_ids]

        sample_list1.extend(sampled_ids)
        from collections import Counter

        occurrences = Counter(sample_list1)
        t_occurrences = {element: 0.95 ** (count - 1) for element, count in occurrences.items()}
        sample_list2 = [t_occurrences[_] for _ in sample_list1]

        sample_list = random.choices(sample_list1, weights=sample_list2, k=sample_n)


    for uni in sample_list:
        df = df_like[df_like['u'] == uni]
        df_un = df_dislike[df_dislike['u'] == uni]

        if len(df) > 1:

            '''
            Listwise Ranking
            '''
            dfp = df_like_p[df_like_p['u'] == uni]
            my_list = dfp['i'].tolist()
            my_list_r = dfp['r'].tolist()

            rndl = [i_ for i_ in range(len(my_list))]
            random.shuffle(rndl)
            my_list = [int(my_list[x]) for x in rndl]
            my_list_r = [int(my_list_r[x]) for x in rndl]

            if len(dfp) > 50:
                topk = 50
            else:
                #topk = len(dfp) - 3
                topk = max(5, len(dfp) - 3)  
            trainlist = my_list[:topk]
            trainlist_r = my_list_r[:topk]

            testlist = my_list[-1:]
            testlist_r = my_list_r[-1:]

            try:
                yy = mf_item[(testlist[0])]
                uu = mf_user[(uni)]
                mf_lable = mf_pred[uu][yy]
            except Exception:
                mf_lable = 'Unknown.'

            historical_interactions = ', '.join([f'"{movie_name_dict[i]}"' for i in trainlist])
            neg_interactions = ', '.join([f'"{movie_name_dict[i]}"' for i in df_un['i'].tolist()[:10]])

            total_list = trainlist[:5]

            total_list_mf = []
            for j_ in total_list:
                try:
                    yy = cm_item[str(j_)]
                    uu = cm_user[str(uni)]
                    mf_label = cm_pred[uu][yy]
                except Exception:
                    mf_label = 1.5
                total_list_mf.append(mf_label)

            total_list_mf_idx = sort_list_reverse_with_indices(total_list_mf)
            total_list_mf_idx = total_list_mf_idx[:5]
            total_list_mf_i = [total_list[k_] for k_ in total_list_mf_idx]

            
            mf_item_sets_ = ', '.join([f'"{movie_name_dict[i]}"' for i in total_list_mf_i])
            true_answer_items_set = ', '.join([f'"{movie_name_dict[i]}"' for i in total_list])






            llike_list = df['i'].tolist()
            if len(df) > 55:
                topkx = 50
            else:
                topkx = len(df) - 3
            trainlist = my_list[:topk]
            like_movie_ids = llike_list[:topkx]
            historical_triples = get_kg_triples_like_1(like_movie_ids, kg_less_test_id_df, kg_less_test_df, entity_id_to_name, relation_id_to_description)
            candidate_triples = get_kg_triples_wait_1(total_list, kg_less_test_id_df, kg_less_test_df, entity_id_to_name, relation_id_to_description)
            like_kg = '; '.join(historical_triples) if historical_triples else 'None.'
            wait_kg = '; '.join(candidate_triples) if candidate_triples else 'None.'



            instruct0 = f'''You are a {kws} recommender system. Your task is to rank a given list of candidate {kws}s based on user preferences and return the top five recommendations.\n\n'''

            instruct1 = f'''User's Liked {kws}s: <historical_interactions>. \nUser's Disliked {kws}s: <user_unpre>\n\nQuestion: How would the user rank the candidate item list: <movie_list> based to historical perference?\n'''
            instruct2 = 'Hint: Another recommender model suggests <cm_result>'
            instruct3 = f'Hint2: These are corresponding entities and relationships for above model’s recommendation for more context information: <like_kg>, <wait_kg>.\n'
            if isHint:
                instruct1 = instruct1 + instruct2

            instruct1 = instruct1.replace('<historical_interactions>', historical_interactions).replace('<user_unpre>', neg_interactions).replace('<movie_list>', true_answer_items_set).replace('<cm_result>', mf_item_sets_)
            instruct3 = instruct3.replace('<like_kg>', like_kg).replace('<wait_kg>', wait_kg)
            instruct2 = '<|endofmessage|><|assistant|>'
            instruct4 = '\n\nPlease only output the top five recommended movies once in the following format:\n1. [Movie Title]\n2. [Movie Title]\n3. [Movie Title]\n4. [Movie Title]\n5. [Movie Title]\n'
            ###fi = {'inst': instruct0 + instruct1 + instruct4 + instruct2}
            #fi = {'inst': instruct0 + instruct1 + instruct3 + instruct4 + instruct2}
            fi = {'messages': [
                {"role": "system", "content": [{"type": "text", "content": ""}]},
                {"role": "user", "content": [{"type": "text", "content": instruct0 + instruct1 + instruct3 + instruct4}]},
                {"role": "assistant",
                 "content": [{"type": "text", "content": 'Answer: ' + str(true_answer_items_set)}]}
            ]}
            mes_list.append(fi)

    with jsonlines.open('./data.jsonl', mode='a') as writer:
        writer.write_all(mes_list)