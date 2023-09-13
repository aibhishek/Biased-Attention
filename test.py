from trainers import vit_trainer, cnn_trainer
import argparse

# parser = argparse.ArgumentParser(description='')
# parser.add_argument('--model', default='vit_l32', type=str,
#                     help='')
# parser.add_argument('--run', default=1, type=int,  help='n')
# parser.set_defaults(bottleneck=True)
# parser.set_defaults(verbose=True)

def main():
    # args = parser.parse_args()
    print("***********Vision Transformers Begin***********")
    vit_trainer.result_generator()
    print("***********Vision Transformer Ends***********")
    print("***********Convolution Neural Networks Begin***********")
    cnn_trainer.result_generator()
    print("***********Convolution Neural Networks Ends***********")

if __name__ == "__main__":
    main()
