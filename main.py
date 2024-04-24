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
    # save_embeddings(df, columns, gpt2_tokenizer, gpt2_model, "embeddings/gpt2_task_embeddings", 1000)
    # save_embeddings(df, columns, bert_tokenizer, bert_model, "embeddings/bert_task_embeddings", 1000)
    
    # gpt2_task_embeddings = load_embeddings("embeddings/gpt2_task_embeddings.pt")
    bert_task_embeddings = load_embeddings("embeddings/bert_task_embeddings.pt")
    
    # Soruya göre insan cevabı bulma
    # task(gpt2_task_embeddings, "soru", "insan cevabı", "GPT2 Question Human")
    task(bert_task_embeddings, "soru", "insan cevabı", "BERT Question Human")
    
    # Soruya göre makine cevabı bulma
    # task(gpt2_task_embeddings, "soru", "makine cevabı", "GPT2 Question Machine")
    task(bert_task_embeddings, "soru", "makine cevabı", "BERT Question Machine")
    
    # İnsan cevabına göre soru bulma
    # task(gpt2_task_embeddings, "insan cevabı", "soru", "GPT2 Human Question")
    task(bert_task_embeddings, "insan cevabı", "soru", "BERT Human Question")
    
    # Makine cevabına göre soru bulma
    # task(gpt2_task_embeddings, "makine cevabı", "soru", "GPT2 Machine Question")
    task(bert_task_embeddings, "makine cevabı", "soru", "BERT Machine Question")
    
    # TSNE Görselleştirme
    # tsne_f(gpt2_task_embeddings, "GPT2")
    tsne_f(bert_task_embeddings, "BERT")
        
if __name__ == "__main__":
    main()
