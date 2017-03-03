library(text2vec)
library(tidytext)
library(stringr)
library(dplyr)
library(Rtsne)
library(dbscan)
library(ggplot2)
library(tm)

data("movie_review")

# paste prevalent bigrams together
stopwords <- tidytext::stop_words$word %>%
  str_replace_all("'", "")

out <- movie_review %>%
  mutate(review = str_replace_all(review, regex("\\<.*?\\>"), " ")) %>%
  unnest_tokens(review, review) %>%
  mutate(review = str_replace_all(review, "'", "")) %>%
  mutate(review = str_replace_all(review, "[^[:alnum:]]", " ")) %>%
  filter(!(review %in% stopwords)) %>%
  group_by(id) %>%
  summarise(review = paste(review, collapse=" ")) %>%
  ungroup() %>%
  unnest_tokens(bigrams, review, token = "ngrams", n=2) %>%
  count(bigrams, sort=TRUE)

#create token iterator
it <- itoken(movie_review$review,
             preprocessor = tolower,
             tokenizer = word_tokenizer,
             ids = movie_review$id,
             progressbar = FALSE)

#set up for word 2 vec (define vocab and vectorizer)
#can't pipe functions for vocab; just use syntax below...... dunno why
vocab       <- create_vocabulary(it, stopwords = tm::stopwords())
vocab_prune <- prune_vocabulary(vocab, term_count_min = 10)
vectorizer  <- vocab_vectorizer(vocab_prune, grow_dtm = FALSE, skip_grams_window = 5L)

#create term-co-occurance matrix (adjacency matrix)
tcm <- create_tcm(it, vectorizer)

#create glove object before fitting tcm to glove
glove <- GlobalVectors$new(word_vectors_size = 50,
                           vocabulary = vocab_prune,
                           x_max = 10)
#fit word vectors
glove$fit(tcm, n_iter = 20)
#extract word vectors
word_vectors <- glove$get_word_vectors()

#function to find closest words to a single word
find_closest_words <- function(search_word, word_vectors_matrix, n=5) {
  search_vec <- word_vectors_matrix[search_word,,drop=FALSE]
  cos_sim    <- text2vec::sim2(x = word_vectors_matrix, y = search_vec, method = "cosine", norm = "l2")

  dplyr::tibble(word = rownames(cos_sim), cos_sim=cos_sim[,1]) %>%
    dplyr::arrange(desc(cos_sim)) %>%
    dplyr::filter(word != search_word) %>%
    head(n=n)
}

#find closest words to words
find_closest_words("movie", word_vectors, n=1)
#   # A tibble: 1 × 2
#      word   cos_sim
#     <chr>     <dbl>
#   1  film 0.8448485

find_closest_words("story", word_vectors, n=1)
#   # A tibble: 1 × 2
#      word   cos_sim
#     <chr>     <dbl>
#   1  plot 0.7263763

# #can also do more complex operations with word vectors
# #needs a lot more training data
# #example
#
# #input
# berlin <- word_vectors["paris", , drop = FALSE] -
#   word_vectors["france", , drop = FALSE] +
#   word_vectors["germany", , drop = FALSE]
# cos_sim = sim2(x = word_vectors, y = berlin, method = "cosine", norm = "l2")
# head(sort(cos_sim[,1], decreasing = TRUE), 5)
#
# #output
# # berlin     paris    munich    leipzig   germany
# # 0.8015347 0.7623165 0.7013252 0.6616945 0.6540700

rtsne_out  <- Rtsne(word_vectors, dims=2, perplexity=30, verbose=TRUE)
dbscan_out <- dbscan(rtsne_out$Y, eps = 1.2, minPts = 3)

cluster_df <- tibble(word    = rownames(word_vectors),
                     cluster = as.character(dbscan_out$cluster),
                     tsne1   = rtsne_out$Y[,1],
                     tsne2   = rtsne_out$Y[,2])

cluster_list <- split(cluster_df$word, cluster_df$cluster)

plot_df <- cluster_df %>%
  filter(cluster != 1)

ggplot(plot_df, aes(x=tsne1, y=tsne2, color=cluster)) +
  geom_point() +
  geom_text(aes(label=word), hjust=0, vjust=0) +
  theme(legend.position = "none")

