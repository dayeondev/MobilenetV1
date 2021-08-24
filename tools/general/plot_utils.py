import matplotlib.pyplot as plt

def show_loss(loss_hist, length):
    
    plt.title('Train Loss')
    plt.plot(range(1, length+1), loss_hist, label='train')
    # plt.plot(range(1, num_epochs+1), loss_hist['val'], label='val')
    plt.ylabel('loss avg')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

def show_accuracy(accu_hist, length, isTrain):
    if isTrain:
        plt.title('Train Accuracy')
    else:
        plt.title('Valid Accuracy')
    plt.plot(range(1, length+1), accu_hist['daisy'], label='daisy')
    plt.plot(range(1, length+1), accu_hist['dandelion'], label='dandelion')
    plt.plot(range(1, length+1), accu_hist['roses'], label='roses')
    plt.plot(range(1, length+1), accu_hist['sunflowers'], label='sunflowers')
    plt.plot(range(1, length+1), accu_hist['tulips'], label='tulips')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
