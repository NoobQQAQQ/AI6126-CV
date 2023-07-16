import argparse
import json
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='../log/1250k.json', type=str)
    args = parser.parse_args()
    print(args)
    train_iters = []
    train_losses = []
    val_iters = []
    val_losses =[]
    with open(args.log, 'r') as f:
        for jsonstr in f.readlines():
            data = json.loads(jsonstr)
            if data['mode'] == 'train':
                train_iters.append(data['iter'])
                train_losses.append(data['loss'])
            else:
                val_iters.append(data['iter'])
                val_losses.append(data['PSNR'])

    plt.subplot(121)
    plt.plot(train_iters, train_losses, label='Loss', linewidth=0.1)
    plt.xlabel('iters')
    plt.ylabel('loss')
    plt.legend(loc="best", fontsize=6)
    plt.title('Train')

    plt.subplot(122)
    plt.plot(val_iters, val_losses, label='PSNR', linewidth=0.5)
    plt.xlabel('iters')
    plt.ylabel('PSNR')
    plt.legend(loc="best", fontsize=6)
    plt.title('Val')
    plt.tight_layout()

    plt.savefig(args.log[:-5] + '.png')
    plt.show()



