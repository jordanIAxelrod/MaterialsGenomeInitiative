import os

import torch.utils.data

import embeddings
import paper_recreation
import neural_network
import DataLoader
import indexation
import Tokenizer
import pandas as pd



def main():
    main_dir = os.path.dirname(__file__)
    train_path = os.path.join(main_dir, '../data/train.tsv')
    test_path = os.path.join(main_dir, '../data/test.tsv')
    train = pd.read_csv(train_path, sep='\t')
    test = pd.read_csv(test_path, sep='\t')
    train.text = train.text.apply(Tokenizer.tokenize_text)
    test.text = test.text.apply(Tokenizer.tokenize_text)

    # Add more preprocessing here


    # continue script
    words, num_words, word_types = indexation.word2idx(list(pd.concat([train, test]).text))
    train_words = words[:len(train)]
    test_words = words[len(train):]
    train['idx'] = train_words
    test['idx'] = test_words
    max_len = max([len(word) for word in words])
    # print(max_len) <--- change to use python logging
    embeddor = embeddings.Embeddings(train.text, test.text)
    experiments = pd.DataFrame(
        {
            'vectorizor':[
                embeddor.TFIDF(),
                embeddor.BOW(),
                embeddor.word2vec(),
                embeddor.TFIDF(),
                embeddor.BOW(),
                embeddor.word2vec(),
                False,
                True
            ],
            'kernel':[
                None,
                None,
                None,
                'rbf',
                'rbf',
                'rbf',
                None,
                None
            ],
            'name':[
                'lr',
                'lr',
                'lr',
                'svm',
                'svm',
                'svm',
                'nn',
                'nn'
            ]
        }
    )
    # create dataloaders for nn
    train_set = DataLoader.TextDataset(train, max_len, num_words)
    test_set = DataLoader.TextDataset(test, max_len, num_words)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)
    nn_embed = neural_network.RNN(4, 50, num_words, 100, False, word_types)
    nn_w2v = neural_network.RNN(4, 50, num_words, 100, False, word_types)
    nn_w2v.set_embedding_weights()
    nn_trainer_e = neural_network.NNTrainer(
        nn_embed,
        5,
        torch.optim.Adam,
        torch.nn.CrossEntropyLoss(),
        train_loader,
        test_loader
    )
    nn_trainer_w = neural_network.NNTrainer(
        nn_embed,
        5,
        torch.optim.Adam,
        torch.nn.CrossEntropyLoss(),
        train_loader,
        test_loader
    )

    paper_replicator = paper_recreation.PaperReplicator(train, test, nn_trainer_e, nn_trainer_w)
    results = paper_replicator.create_results(experiments)
    results.to_csv(os.path.join(main_dir, '../results/results.csv'))
if __name__=='__main__':
    main()

