class Config:
    batchSize = 64
    numClasses = 2
    numDimensions = 300
    learning_rate = 1e-4
    drop_keep = 0.75
    lstm_layers = 3
    maxSeqLength = 20
    evaluate_every = 10
    n_epoch = 200
    gradient_clip_norm = 1e2
