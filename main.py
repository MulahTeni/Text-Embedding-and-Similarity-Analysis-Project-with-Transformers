# Bir soruya en benzeyen 5 insan cevabını bulup gerçek cevaba göre top1 ve top5 başarılarını hesaplamak. rasgele 1000 soru üzerinden.
# Bir soruya en benzeyen 5 makine cevabını bulup gerçek cevaba göre top1 ve top5 başarılarını hesaplamak. rasgele 1000 soru üzerinden.
# Bir insan cevabına en benzeyen 5 soruyu bulup gerçek soruya göre top1 ve top5 başarılarını hesaplamak. rasgele 1000 insan cevabı üzerinden.
# Bir makine cevabına en benzeyen 5 soruyu bulup gerçek soruya göre top1 ve top5 başarılarını hesaplamak. rasgele 1000 makine cevabı üzerinden.

# Soru, insan cevabı, makine cevabı bu temsiller üzerine tsne uygulanarak \
#   2 boyutta türlerine göre renklendirilerek görselleştirilecektir.
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from task_functions import tasks as task, tsne_f, read_excel_file, save_embeddings, load_embeddings

def main():
    # gpt2
    # gpt2 = "ytu-ce-cosmos/turkish-gpt2-large-750m-instruct-v0.1"
    # gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2)
    # gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2)

    # bert
    # bert = "ytu-ce-cosmos/turkish-base-bert-uncased"
    # bert_tokenizer = AutoTokenizer.from_pretrained(bert)
    # bert_model = AutoModel.from_pretrained(bert)

    # data
    # excel_path = "soru_cevap.xlsx"
    # df = read_excel_file(excel_path)
    # columns = df.columns
    
    # Yerel dosyalara çıkartılan embedingleri kaydet
    # save_embeddings(df, columns, gpt2_tokenizer, gpt2_model, "gpt2_task_embeddings", 1000)
    # save_embeddings(df, columns, bert_tokenizer, bert_model, "bert_task_embeddings", 1000)
    
    # gpt2_task_embeddings = load_embeddings("gpt2_task_embeddings.pt")
    bert_task_embeddings = load_embeddings("bert_task_embeddings.pt")
    
    # Soruya göre insan cevabı bulma
    # task(gpt2_task_embeddings, "soru", "insan cevabı", "Question_Human_GPT2")
    task(bert_task_embeddings, "soru", "insan cevabı", "Question_Human_BERT")
    
    # Soruya göre makine cevabı bulma
    # task(gpt2_task_embeddings, "soru", "makine cevabı", "Question_Machine_GPT2")
    task(bert_task_embeddings, "soru", "makine cevabı", "Question_Machine_BERT")
    
    # İnsan cevabına göre soru bulma
    # task(gpt2_task_embeddings, "insan cevabı", "soru", "Human_Question_GPT2")
    task(bert_task_embeddings, "insan cevabı", "soru", "Human_Question_BERT")
    
    # Makine cevabına göre soru bulma
    # task(gpt2_task_embeddings, "makine cevabı", "soru", "Machine_Question_GPT2")
    task(bert_task_embeddings, "makine cevabı", "soru", "Machine_Question_BERT")
    
    # TSNE Görselleştirme
    # tsne_f(gpt2_task_embeddings, "TSNE_GPT2")
    tsne_f(bert_task_embeddings, "TSNE_BERT")
        
if __name__ == "__main__":
    main()
