import matplotlib.pyplot as plt

def show_plot(hist, length, isTrain):
    
    if isTrain:
        plt.title('Train Loss')
    else:
        plt.title('Valid Loss')
    plt.plot(range(0, length), hist['train'], label='train')
    plt.plot(range(0, length), hist['valid'], label='valid')
    plt.ylabel('loss avg')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

def show_accuracy(accu_hist, length, isTrain):
    if isTrain:
        plt.title('Train Accuracy')
    else:
        plt.title('Valid Accuracy')
    plt.plot(range(0, length), accu_hist['daisy'], label='daisy')
    plt.plot(range(0, length), accu_hist['dandelion'], label='dandelion')
    plt.plot(range(0, length), accu_hist['roses'], label='roses')
    plt.plot(range(0, length), accu_hist['sunflowers'], label='sunflowers')
    plt.plot(range(0, length), accu_hist['tulips'], label='tulips')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


