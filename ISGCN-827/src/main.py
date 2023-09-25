"""SGCN runner."""

from isgcn import SignedGCNTrainer
from param_parser import parameter_parser
from utils import tab_printer, read_graph, score_printer, save_logs

def main():
    """
    Parsing command line parameters.
    Creating target matrix.
    Fitting an SGCN.
    Predicting edge signs and saving the embedding.
    """
    args = parameter_parser()   #  0.01 seed=7 [128,128,128]  seed[42,0,5,7,11]
    tab_printer(args)

    args.edge_path = '../input/OTC.csv'   # Alpha、OTC、epinions、Slashdot：[128,128],[128,128,128]
    args.embedding_path = '../output/embedding/OTC_sgcn.csv' # epinions、Slashdot
    # args.features_path = './input/bitcoin_otc.csv'
    args.regression_weights_path = '../output/weights/OTC_sgcn.csv'
    args.epochs = 400

    edges = read_graph(args)  # 导入训练数据
    trainer = SignedGCNTrainer(args, edges)
    trainer.setup_dataset()  # 计算特征
    trainer.create_and_train_model()
    if args.test_size > 0:
        trainer.save_model()
        score_printer(trainer.logs)
        save_logs(args, trainer.logs)

if __name__ == "__main__":
    main()
