
import os
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from utils.utils import get_learner, get_data_dir
from utils.args import TrainArgumentsManager

from train import init_clients

arguments_manager = TrainArgumentsManager()
arguments_manager.parse_arguments()
args_ = arguments_manager.args

data_dir = get_data_dir(args_.experiment)

if "logs_dir" in args_:
    logs_dir = args_.logs_dir
else:
    logs_dir = os.path.join("logs", arguments_manager.args_to_string())

if "chkpts_dir" in args_:
    chkpts_dir = args_.chkpts_dir
else:
    chkpts_dir = None

clients = \
        init_clients(
            args_,
            data_dir=os.path.join(data_dir, "train"),
            logs_dir=os.path.join(logs_dir, "train"),
            chkpts_dir=os.path.join(chkpts_dir, "train") if chkpts_dir else None
        )


train_datasets = [client.train_iterator.dataset for client in clients]
val_datasets = [client.val_iterator.dataset for client in clients]

global_train_dataset = ConcatDataset(train_datasets)
global_val_dataset = ConcatDataset(val_datasets)

global_train_loader = DataLoader(global_train_dataset, batch_size=args_.bz, shuffle=True)
global_val_loader = DataLoader(global_val_dataset, batch_size=args_.bz, shuffle=False)

global_learner = \
    get_learner(
        name=args_.experiment,
        model_name=args_.model_name,
        device=args_.device,
        optimizer_name=args_.optimizer,
        scheduler_name=args_.lr_scheduler,
        initial_lr=args_.lr,
        n_rounds=args_.n_rounds,
        seed=args_.seed,
        input_dimension=args_.input_dimension,
        hidden_dimension=args_.hidden_dimension,
        mu=args_.mu
    )

loss, metric = global_learner.evaluate_iterator(iterator=global_val_loader)

print(f"Round {0}, loss {loss}, accuracy {metric}")

for ii in tqdm(range(args_.n_rounds)):

    global_learner.fit_epochs(
        iterator=global_train_loader,
        n_epochs=1
    )

    loss, metric = global_learner.evaluate_iterator(iterator=global_val_loader)

    print(f"Round {ii}, loss {loss}, accuracy {metric}")