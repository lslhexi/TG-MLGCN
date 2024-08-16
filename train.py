import argparse

from torch.utils.data import DataLoader
from tgmodels import *
from tgdata import *
from util import *
from eval_metric import *

def get_parser():
    parser = argparse.ArgumentParser(description='WILDCAT Training')
    parser.add_argument('--data',default='',help='path to dataset (e.g. data/')
    parser.add_argument('--image_size', '-i', default=448, type=int, help='resize image size')
    parser.add_argument('-j', '--workers', default=4, type=int,help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',help='number of total epochs to run')
    parser.add_argument('-b', '--batchsize', default=8, type=int,help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float, help='learning rate for pre-trained layers')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--checkpoint_interval', '--ci', default=5, type=int, help='interval of saving .pth')
    parser.add_argument('--log_interval', '--li', default=10, type=int, help='interval of log')
    parser.add_argument('--val_interval', '--vi', default=1, type=int, help='interval of validation')
    parser.add_argument('--MultiStepLR', '--msl', default=None, type=bool, help='lr_decay')
    return parser.parse_args()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global args
    args = get_parser()

    root=args.data

    train_dataset = TGdata(root, img_size=args.image_size,phase='train', inp_name=r'word_embding\word_embding.npy')
    val_dataset = TGdata(root, img_size=args.image_size,phase='test', inp_name=r'word_embding\word_embding.npy')
    num_classes = 20

    model = TGGCNResnet(num_classes=num_classes, t=0.4, adj_file=r'data\TG1\anno\train_no_rpt.json')
    model = model.to(device)
    
    # define loss function
    loss_fn = nn.MultiLabelSoftMarginLoss()
    loss_fn = loss_fn.cuda()
    
    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.MultiStepLR:
        milestones = [args.epochs - 3]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=args.batchsize,shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batchsize, shuffle=False)

    # epoch
    num_epochs = args.epochs
    checkpoint_interval=args.checkpoint_interval
    log_interval=args.log_interval
    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(inputs[0],inputs[1])
            labels=labels.to(device)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if args.MultiStepLR:
                scheduler.step()
            if i % log_interval == 0:  # 每log_interval打印一次损失
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{len(train_dataloader)}], Loss: {running_loss / 10:.4f}'+',learning_rate: {}'.format(optimizer.param_groups[0]['lr']))
                running_loss = 0.0
        # eval
        if epoch % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                all_outputs = []
                all_labels = []
                for i , (inputs, labels) in enumerate(val_dataloader):
                    outputs = model(inputs[0],inputs[1])
                    labels = labels.to(device)
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item()
                    all_outputs.append(outputs.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
                    if i % log_interval == 0:  # 每1log_interval打印一次损失
                        print(f'val_Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{len(val_dataloader)}]')
                all_outputs = np.vstack(all_outputs)
                all_labels = np.vstack(all_labels)
                precision, recall, f1 = precision_recall_f1_at_k(all_outputs, all_labels, k=5)
                print(f'Validation Loss after Epoch {epoch + 1}: {val_loss / len(val_dataset):.4f}')
                print(f'Precision@5: {precision:.4f}, Recall@5: {recall:.4f}, F1-Score@5: {f1:.4f}')

        # 保存模型
        if epoch % checkpoint_interval ==0:
            torch.save(model.state_dict(), 'checkpoint/TG/model_weights.pth')

if __name__ == '__main__':
    main()
