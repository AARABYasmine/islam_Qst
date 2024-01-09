import torch
import numpy as np



# Cleaning Pipeline
def nlp_pipeline(answer):
    answer = remove_arabic_diacritics(answer)
    answer = remove_punctuations(answer)
    answer_tokens = word_tokenize(answer)
    answer_tokens_without_stop_words = remove_stop_words(answer_tokens)
    answer_stemming_tokens = stemming(answer_tokens)
    answer_lemma_tokens = lemmatization(answer_tokens)

    return ' '.join(answer_tokens)


def concatenate_tfidf_fastText(answer, documents):
    tfidf_rep = get_tf_idf(answer, documents)
    fasttext_rep = get_fastText_representation(answer)

    # Normalize representations
    tfidf_rep_normalized = normalize_vector(tfidf_rep)
    fasttext_rep_normalized = normalize_vector(fasttext_rep)

    return np.concatenate([tfidf_rep_normalized, fasttext_rep_normalized])


def Concat_idx_tensor(X_tensor, indices):
    # idx_tensor contains the indices of sentences
    idx_tensor = torch.tensor(indices)
    # Concatenate the idx_tensor with X_tensor along the second dimension (dim=1)
    return torch.cat((idx_tensor.unsqueeze(1).float(), X_tensor), dim=1)


def preprocess_data(question_idx, answer, documents):
    answer = nlp_pipeline(answer)
    concatenated_rep = concatenate_tfidf_fastText(answer, documents).tolist()
    x_val = np.array([concatenated_rep])
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    x_val_tensor = Concat_idx_tensor(x_val_tensor, [question_idx])
    return None