from duplicate_question_pairs.preprocess.preprocessor import preprocess


def main():
    file_path = "data/train_small.csv"
    X_test, Y_test, embed_construct = preprocess(file_path, is_train=False)


if __name__ == '__main__':
    main()