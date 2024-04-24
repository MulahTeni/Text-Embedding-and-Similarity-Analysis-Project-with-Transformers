from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.spatial.distance import cosine
import numpy, random, heapq, torch, logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MaxHeap:
    def __init__(self, capacity=5):
        self.heap = []
        self.capacity = capacity

    def add(self, score, index):
        entry = (score, index)
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, entry)
        else:
            heapq.heappushpop(self.heap, entry)

    def get_top_scores(self):
        sorted_heap = sorted(self.heap, reverse=True)
        scores = [score for score, index in sorted_heap]
        indexes = [index for score, index in sorted_heap]
        return scores, indexes

def read_excel_file(file_path):
    return pd.read_excel(file_path, usecols=[0, 1, 2])

def save_embeddings(df, columns, tokenizer, model, file_path, limit):
    sampled_df = df.sample(n=limit)
    embeddings_dict = {col: {} for col in columns}
    
    for index, row in sampled_df.iterrows():
        print(f"Calculating embeddings for row {index} for {file_path} file")
        logging.info(f"Calculating embeddings for row {index} for {file_path} file")
        for col in columns:
            try:
                text = str(row[col])
                inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states
                    last_layer_hidden_state = hidden_states[-1]
                    embedding = last_layer_hidden_state.mean(dim=1).squeeze().detach()
                embeddings_dict[col][index] = embedding.cpu().numpy()
            except Exception as e:
                print(f"Failed to process row {index}: {e}")
                logging.error(f"Failed to process row {index}: {e}")
                
    torch.save(embeddings_dict, f"{file_path}.pt")

def load_embeddings(file_path):
    embeddings_dict = torch.load(file_path)
    return embeddings_dict

def calculate_cosine_similarity(src_embedding, dest_embedding):
    if len(src_embedding.shape) > 1:
        src_embedding = src_embedding.squeeze()
    if len(dest_embedding.shape) > 1:
        dest_embedding = dest_embedding.squeeze()
    
    return 1 - cosine(src_embedding, dest_embedding)


def check_heap_tops(heap_indexes, real_index):
    for i in range(len(heap_indexes)):
        if heap_indexes[i] == real_index:
            return i
    return -1

def visualize_and_save(data, title_base, file_name_base, not_found, first_place, elsewhere):
    positions = [i+1 for i in range(len(data))]
    scores = [x[1] for x in data]
    try:
        plt.figure(figsize=(45, 8))
        plt.plot(positions, scores, marker='o', linestyle='-', color='b')
        plt.title(f'{title_base} - Cosine Similarity Scores')
        plt.xlabel('Random 1000 Example')
        plt.ylabel('Max or Correct Cosine Similarity Score')
        plt.grid(True)
        plt.savefig(f'{file_name_base} Cosine.png')
        plt.show()
    except Exception as e:
        logging.error(f"Failed to generate or save plot: {e}")
    try:
        indexes = sorted([x[0] for x in data])
        plt.figure(figsize=(35, 8))
        plt.plot(positions, indexes, marker='o', linestyle='-', color='r')
        plt.title(f'{title_base} - Indexes')
        plt.xlabel('Random 1000 Example')
        plt.ylabel('Index in Heap or 0')
        plt.grid(True)
        plt.savefig(f'{file_name_base} Index.png')
        plt.show()
    except Exception as e:
        logging.error(f"Failed to generate or save plot: {e}")
    
    try:
        counts = [not_found, first_place, elsewhere]
        labels = ['Not found', 'TOP 1', 'TOP 5']
        colors = ['grey', 'green', 'orange']
        explode = (0.01, 0.01, 0.01)
        
        plt.figure(figsize=(8, 8))
        plt.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=0, explode=explode)
        plt.title('Distribution of Query Results')
        plt.savefig(f'{file_name_base} Pie Graph.png')
        plt.show()
    except Exception as e:
        logging.error(f"Failed to generate or save plot: {e}")

def tasks(embeddings, src_text, dest_text, title):
    keys_list = list(embeddings[src_text].keys())
    data_len = len(keys_list)
    model_cht, not_found, first_place, elsewhere = [], 0, 0, 0
    
    for i in range(data_len):
        print(f"Calculating similarity scores for {title}, loop: {i}")
        logging.info(f"Calculating similarity scores for {title}, loop: {i}")
        question_key = keys_list[i]
        question = embeddings[src_text][question_key]
        model_heap = MaxHeap()
        
        for answer_key in embeddings[dest_text].keys():
            current_answer = embeddings[dest_text][answer_key]
            similarity = calculate_cosine_similarity(question, current_answer)
            model_heap.add(similarity, answer_key)

        top_scores = model_heap.get_top_scores()
        
        if isinstance(top_scores, tuple) and len(top_scores) == 2:
            scores, indices = top_scores
            model_index = next((i for i, index in enumerate(indices) if index == question_key), -1)
        else:
            continue
            
        if model_index == -1:
            not_found += 1
            model_cht.append((0, 0))
        else:
            if model_index == 0:
                first_place += 1
            elsewhere += 1
            model_cht.append((model_index + 1, scores[model_index]))
    
    visualize_and_save(model_cht, title, f'{title} Visualization', not_found, first_place, elsewhere)

def tsne_f(embeddings, title):
    try:
        embeddings = pd.DataFrame(embeddings)
        cols = embeddings.columns
        
        all_embeddings = []
        labels = []
        
        for category in cols:
            print(f"Calculating TSNE for {title}")
            logging.info(f"Calculating TSNE for {title}")
            
            if category in embeddings.columns:
                category_embeddings = numpy.array(embeddings[category].tolist())
                new_shape = category_embeddings.shape[2]
                
                category_embeddings = category_embeddings.reshape(-1, new_shape)
                
                all_embeddings.append(category_embeddings)
                labels.extend([category] * len(category_embeddings))
                
    except Exception as e:
        printf(f"TSNE error: {e}")
        logging.error(f"TSNE error: {e}")
        return

    if not all_embeddings:
        print("No embeddings available to visualize.")
        return

    try:
        all_embeddings = numpy.concatenate(all_embeddings)
    except ValueError as e:
        print("Error concatenating embeddings:", e)
        return
        
    perplexity_value = min(30, len(all_embeddings) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
    reduced_embeddings = tsne.fit_transform(all_embeddings)
    
    plt.figure(figsize=(10, 8))
    for category in set(labels):
        indices = [i for i, label in enumerate(labels) if label == category]
        plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], label=category, alpha=0.7)
    
    plt.legend()
    plt.title(f't-SNE Visualization of {title}')
    plt.xlabel('t-SNE axis 1')
    plt.ylabel('t-SNE axis 2')
    plt.savefig(f'{title} TSNE.png', format='png', dpi=300)
    plt.show()

logger = logging.getLogger(__name__)