"""
module adapted from https://github.com/kevinzakka/one-shot-siamese
"""
import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='Siamese Network')


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--valid_trials', type=int, default=320,
                      help='# of validation 1-shot trials')
data_arg.add_argument('--test_trials', type=int, default=400,
                      help='# of test 1-shot trials')
data_arg.add_argument('--way', type=int, default=20,
                      help='Ways in the 1-shot trials')
data_arg.add_argument('--num_train', type=int, default=90000,
                      help='# of images in train dataset')
data_arg.add_argument('--batch_size', type=int, default=64,
                      help='# of images in each batch of data')
data_arg.add_argument('--num_workers', type=int, default=1,
                      help='# of subprocesses to use for data loading')
data_arg.add_argument('--shuffle', type=str2bool, default=True,
                      help='Whether to shuffle the dataset between epochs')
data_arg.add_argument('--augment', type=str2bool, default=True,
                      help='Whether to use data augmentation for train data')


# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--is_train', type=str2bool, default=True,
                       help='Whether to train or test the model')
train_arg.add_argument('--epochs', type=int, default=200,
                       help='# of epochs to train for')
train_arg.add_argument('--init_momentum', type=float, default=0.5,
                       help='Initial layer-wise momentum value')
train_arg.add_argument('--lr_patience', type=int, default=1,
                       help='Number of epochs to wait before reducing lr')
train_arg.add_argument('--train_patience', type=int, default=20,
                       help='Number of epochs to wait before stopping train')


# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--flush', type=str2bool, default=False,
                      help='Whether to delete ckpt + log files for model no.')
misc_arg.add_argument('--num_model', type=int, default=1, required=True,
                      help='Model number used for unique checkpointing')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True,
                      help="Whether to run on the GPU")
misc_arg.add_argument('--best', type=str2bool, default=True,
                      help='Load best model or most recent for testing')
misc_arg.add_argument('--random_seed', type=int, default=1,
                      help='Seed to ensure reproducibility')
misc_arg.add_argument('--data_dir', type=str, default='./data/changed/',
                      help='Directory in which data is stored')
misc_arg.add_argument('--plot_dir', type=str, default='./plots/',
                      help='Directory in which plots are stored')
misc_arg.add_argument('--ckpt_dir', type=str, default='./ckpt/',
                      help='Directory in which to save model checkpoints')
misc_arg.add_argument('--logs_dir', type=str, default='./logs/',
                      help='Directory in which logs wil be stored')
misc_arg.add_argument('--resume', type=str2bool, default=False,
                      help='Whether to resume training from checkpoint')


class Configuration():
    
    def __init__(self, valid_trials=320, test_trials=400, way=20,
                 num_train=90000, batch_size=128, num_workers=1,
                 shuffle=True, augment=True, is_train=True,
                 epochs=200, init_momentum=0.5, lr_patience=1,
                 train_patience=20, flush=False, num_model=1,
                 use_gpu=True, best=True, random_seed=1,
                 data_dir='./data/changed/', plot_dir='./plots/',
                 ckpt_dir='./ckpt/', logs_dir='./logs/', resume=False):
        self.valid_trials = valid_trials
        self.test_trials = test_trials
        self.way = way
        self.num_train = num_train
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.augment = augment
        self.is_train = is_train
        self.epochs = epochs
        self.init_momentum = init_momentum
        self.lr_patience = lr_patience
        self.train_patience = train_patience
        self.flush = flush
        self.num_model = num_model
        self.use_gpu = use_gpu
        self.best = best
        self.random_seed = random_seed
        self.data_dir = data_dir
        self.plot_dir = plot_dir
        self.ckpt_dir = ckpt_dir
        self.logs_dir = logs_dir
        self.resume = resume
        
        self._create_fields_help()
    
    
    def _create_fields_help(self):
        help_dict = dict(valid_trials='# of validation 1-shot trials',
                         test_trials='# of test 1-shot trials',
                         way='Ways in the 1-shot trials',
                         num_train='# of images in train dataset',
                         batch_size='# of images in each batch of data',
                         num_workers='# of subprocesses to use for data loading',
                         shuffle='Whether to shuffle the dataset between epochs',
                         augment='Whether to use data augmentation for train data',
                         is_train='Whether to train or test the model',
                         epochs='# of epochs to train for',
                         init_momentum='Initial layer-wise momentum value',
                         lr_patience='Number of epochs to wait before reducing lr',
                         train_patience='Number of epochs to wait before stopping train',
                         flush='Whether to delete ckpt + log files for model no.',
                         num_model='Model number used for unique checkpointing',
                         use_gpu="Whether to run on the GPU",
                         best='Load best model or most recent for testing',
                         random_seed='Seed to ensure reproducibility',
                         data_dir='Directory in which data is stored',
                         ckpt_dir='Directory in which to save model checkpoints',
                         logs_dir='Directory in which logs wil be stored',
                         resume='Whether to resume training from checkpoint')
        self.help_dict = help_dict
        
        
    def help(self, field=None):
        if field:
            print(f"{field} : {self.help_dict[field]}")
        else:
            for field in self.help_dict.keys():
                print(f"{field} : {self.help_dict[field]}")
    
    def print_values(self):
        for field in self.help_dict.keys():
            value = getattr(self, field)
            print(f"{field} : {value}")
        
def get_config():
    #config, unparsed = parser.parse_known_args()
    
    unparsed = None
    config = Configuration()
    return config, unparsed