from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.models import resnet18
import numpy as np
import os
import argparse


parser = argparse.ArgumentParser(description="Training a FaceNet facial recognition model")
parser.add_argument('--dataroot', '-d', type=str, required=True,
                    help="(REQUIRED) Absolute path to the dataset folder"
                    )
parser.add_argument('--epochs', default=150, type=int,
                    help="Required training epochs (default: 150)"
                    )
parser.add_argument('--model', type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "inceptionresnetv1", "mobilenetv2"],
                    help="The required model architecture for training: ('resnet18','resnet34', 'resnet50', 'resnet101', 'resnet152', 'inceptionresnetv1', 'mobilenetv2'), (default: 'resnet18')"
                    )
parser.add_argument('--batch_size', default=32, type=int,
                    help="Batch size (default: 32)"
                    )
parser.add_argument('--seed', default=320, type=int,
                    help="random seed (default: 0)"
                    )
parser.add_argument('--lr', default=0.1, type=float,
                    help="Learning rate for the optimizer (default: 0.1)"
                    )
parser.add_argument('--dataset', default='vggface2',  type=str,
                    help='vggface2 / webface'
                    )
args = parser.parse_args()
data_dir = args.dataroot
torch.manual_seed(320)
np.random.seed(320)

batch_size = args.batch_size
epochs = args.epochs
workers = 0 if os.name == 'nt' else 8

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

if not os.path.exists(data_dir + "_aligned"):
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))
    dataset.samples = [
        (p, p.replace(data_dir, data_dir + '_aligned'))
            for p, _ in dataset.samples
    ]

    loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        collate_fn=training.collate_pil
    )

    for i, (x, y) in enumerate(loader):
        mtcnn(x, save_path=y)
        print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')

    # Remove mtcnn to reduce GPU memory usage
    del mtcnn
    
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

trainset = datasets.ImageFolder(data_dir + '_aligned', transform=transform_train)
testset = datasets.ImageFolder(data_dir + '_aligned', transform=transform_test)
img_inds = np.arange(len(trainset))
np.random.shuffle(img_inds)
train_inds = img_inds[:int(0.8 * len(img_inds))]
val_inds = img_inds[int(0.8 * len(img_inds)):]

torch.manual_seed(args.seed)
np.random.seed(args.seed)
train_loader = DataLoader(
    trainset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(train_inds)
)
val_loader = DataLoader(
    testset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(val_inds)
)

loss_fn = torch.nn.CrossEntropyLoss()
metrics = {
    'fps': training.BatchTimer(),
    'acc': training.accuracy
}
if args.model == "inceptionresnetv1":
    model = InceptionResnetV1(
        classify=True,
        num_classes=len(trainset.class_to_idx)
    ).to(device)
elif args.model == "resnet18":
    model = resnet18(num_classes=len(trainset.class_to_idx)).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = MultiStepLR(optimizer, [5, 10])

os.makedirs("checkpoint", exist_ok=True)
os.makedirs("checkpoint/{}".format(args.dataset), exist_ok=True)
os.makedirs("checkpoint/{}/logs".format(args.dataset), exist_ok=True)
os.makedirs("checkpoint/{}/models".format(args.dataset), exist_ok=True)

writer = SummaryWriter(log_dir="checkpoint/{}/logs/{}_{}_{}_{}.txt".format(args.dataset, args.model, args.batch_size, args.lr, args.seed))
writer.iteration, writer.interval = 0, 10

print('\n\nInitial')
print('-' * 10)
model.eval()
training.pass_epoch(
    model, loss_fn, val_loader,
    batch_metrics=metrics, show_running=True, device=device,
    writer=writer
)

first_drop, second_drop = False, False

for epoch in range(epochs):
    if (not first_drop) and (epoch+1) >= 0.5 * args.epochs:
        for g in optimizer.param_groups:
            g['lr'] *= 0.1
        first_drop = True
    if (not second_drop) and (epoch+1) >= 0.75 * args.epochs:
        for g in optimizer.param_groups:
            g['lr'] *= 0.1
        second_drop = True
    print('\nEpoch {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)

    model.train()
    training.pass_epoch(
        model, loss_fn, train_loader, optimizer, scheduler,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    model.eval()
    training.pass_epoch(
        model, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )
    
    state = {"epoch": epoch, "model": model.state_dict()}
    torch.save(state, "checkpoint/{}/models/{}_{}_{}_{}.txt".format(args.dataset, args.model, args.batch_size, args.lr, args.seed))

writer.close()
