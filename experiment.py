"""
a tool to run an experiment with
    - a given Net class and params,
    - optimizer class and params,
    - learning rate scheduler,
    - train data with splits,
    - test data,
    - loss functions and metrics

it needs to
    - run training on all folds with validation,
    - calculate and keep losses and metrics
    - save/restore from checkpoints
    - save and keep final trained models
    - use trained models to calculate and save predictions
    -
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data.dataloader import default_collate, DataLoader

from tqdm import tqdm
import time

from comp_utils import to_rle

import uuid
import os
import pickle
import glob
import json


class FnValueTracker(object):
    """
    Object that can be called, and calculated values would be stored internally

    - For each call a batch and epoch number must be given so the values would be stored appropriately.
    - Already calculated values can be added without calling the object
    - The values can then be requested in different ways: all, aggregated by epoch, last epoch only
    """
    def __init__(self, fn):
        self.fn = fn
        self.reset()

    def reset(self):
        self.values = []
        self.df = None

    def __call__(self, batch_i, epoch_i, *args, **kwargs):
        value = self.fn(*args, **kwargs).mean().item()
        self.add_value(batch_i, epoch_i, value)

    def add_value(self, batch_i, epoch_i, value):
        self.df = None
        self.values.append({
            'batch_i': batch_i,
            'epoch_i': epoch_i,
            'value': value
        })

    def get_all_values(self):
        if self.df is None:
            self.df = pd.DataFrame(self.values)
        return self.df.copy()

    def get_last_epoch_mean(self):
        if self.df is None:
            self.df = pd.DataFrame(self.values)
        if len(self.df):
            max_ep = self.df['epoch_i'].max()
            return self.df[self.df['epoch_i']==max_ep]['value'].mean()
        else:
            raise Exception("did not add any value yet")

    def get_all_epoch_means(self):
        if self.df is None:
            self.df = pd.DataFrame(self.values)
        if len(self.df):
            return self.df.groupby('epoch_i')['value'].mean()
        else:
            raise Exception("did not add any value yet")


class Experiment(object):
    """
    need to store
        - each fold model
        - each fold train predictions
        - each fold test predictions
        - each fold train loss and metric for all epochs
        - each fold valid loss and metric for all epochs

    Saves directories
    experiments/
        name/
            ...
            models/model_x/checkpoint_y
            ...



    """
    def __init__(self, ModelClass, model_params:dict, OptClass, opt_params:dict, loss_fn:callable, splits,
                 train_dataset, test_dataset, LrSchClass=None, lr_sch_params:dict=None, lr_metric:str=None,
                 train_metrics:dict=None, val_metrics:dict=None, pred_post_process:callable=None,
                 device='cpu', max_iters=10, batch_size=64, name=None, checkpoint_every=50, additional_meta=None):
        """
        Instantiate an experiment
        Parameters
        ----------
        ModelClass
            neural net class to use in the experiment

        model_params
            will be passed as kwargs when ModelClass is instantiated

        OptClass
            Optimizer class

        opt_params
            will be passed as kwargs when OptClass is instantiated

        loss_fn
            callable that will be used for calculating loss during training

        splits
            N folds of indices for training and validation parts of the training dataset

        train_dataset
            train pytorch dataset

        test_dataset
            test pytorch dataset

        LrSchClass
            optional, learning rate scheduler class

        lr_sch_params
            optional, will be passed as kwargs when LrSchClass is instantiated

        lr_metric
            optional, name of the metric to be used with learning rate scheduler,
            the metric must exist in the given `val_metrics`

        train_metrics
            optional, dict of {name:fn}, for each new training process and for each given
            metric function an instance of `FnValueTracker` will be created to keep track of values
            These metrics will be calculated using training data.

        val_metrics
            optional, same as train_metrics, but values will be calculated using validation data

        pred_post_process
            optional, function to call on predicted values, it will be called for each
            prediction, train, validation or test

        device
            device to use in pytorch

        max_iters
            max number of iterations to train each fold

        batch_size
            batch size to use for training or inference

        name
            optional, name of model, will be used as a directory name to store models and predictions
            If not given, a random string will be generated

        checkpoint_every
            how often store checkpoints during training

        additional_meta
            additional key-value info to save alongside the training/inference results of the experiment
        """
        self.ModelClass = ModelClass
        self.model_params = model_params

        self.OptClass = OptClass
        self.opt_params = opt_params

        self.LrSchClass = LrSchClass
        self.lr_sch_params = lr_sch_params
        self.lr_metric = lr_metric

        self.loss_fn = loss_fn

        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.pred_post_process = pred_post_process

        self.splits = splits
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.max_iters = max_iters
        self.batch_size = batch_size
        self.device = device
        self.checkpoint_every = checkpoint_every
        self.name = str(uuid.uuid4()) if name is None else name

        # create data loaders for whole of training and test datasets to be used for generating predictions
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False,
                                       pin_memory=self.device=='cuda',
                                       sampler=torch.utils.data.sampler.SequentialSampler(train_dataset),
                                       num_workers=2)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                      pin_memory=self.device=='cuda',
                                      sampler=torch.utils.data.sampler.SequentialSampler(test_dataset),
                                      num_workers=2)

        self.meta = {
            'ModelClass': '{}.{}'.format(ModelClass.__module__, ModelClass.__name__),
            'model_params': model_params,
            'OptClass': '{}.{}'.format(OptClass.__module__, OptClass.__name__),
            'opt_params': opt_params,
            'LrSchClass': None if LrSchClass is None else '{}.{}'.format(LrSchClass.__module__, LrSchClass.__name__),
            'lr_sch_params': opt_params,
            'lr_metric': lr_metric,
            'loss_fn': '{}.{}'.format(loss_fn.__module__, getattr(loss_fn, '__name__', str(loss_fn))),
            'max_iters': max_iters,
            'batch_size': batch_size,
            'results': [],
            'submissions': []
        }
        if additional_meta is not None:
            self.meta.update(additional_meta)

        self.experiment_dir = os.path.join('experiments', self.name)
        os.makedirs(self.experiment_dir, exist_ok=True)

    def _make_loaders(self, tr_ix, va_ix):
        """Helper to make new train and validation data loaders for a fold"""
        tr_set = Subset(self.train_dataset, tr_ix)
        val_set = Subset(self.train_dataset, va_ix)

        tr_loader = DataLoader(tr_set, batch_size=self.batch_size, shuffle=False,
                               pin_memory=(self.device == 'cuda'), num_workers=5,
                               sampler=torch.utils.data.sampler.WeightedRandomSampler(torch.ones(len(tr_ix)),
                                                                                      len(tr_ix)))

        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False,
                                pin_memory=(self.device == 'cuda'),
                                sampler=torch.utils.data.sampler.SequentialSampler(val_set), num_workers=2)

        return tr_loader, val_loader

    def _add_metrics_to_meta(self, fold_i, trainer):
        """add metrics final epoch values, and all epoch means to the metadata of the experiment"""
        result = {
            'name': self.name,
            'fold': fold_i
        }
        for name, obj in trainer.train_metrics.items():
            result['train_' + name] = obj.get_last_epoch_mean()
            result['train_' + name + '_all'] = obj.get_all_epoch_means().tolist()
        for name, obj in trainer.val_metrics.items():
            result['val_' + name] = obj.get_last_epoch_mean()
            result['val_' + name + '_all'] = obj.get_all_epoch_means().tolist()
        self.meta['results'].append(result)

        # save to metadata
        with open(os.path.join(self.experiment_dir, 'metadata.json'), 'w') as f:
            json.dump(self.meta, f)

    def _make_predictions(self, fold_i, trainer):
        """
        Helper to make predictions using a one fold model for the whole
        training and test datasets and store to files
        """
        with open(os.path.join(self.experiment_dir, 'train_predictions_{}.pickle'.format(fold_i)), 'wb') as f:
            pickle.dump(trainer.predict(self.train_loader, has_target=True), f)
        with open(os.path.join(self.experiment_dir, 'test_predictions_{}.pickle'.format(fold_i)), 'wb') as f:
            pickle.dump(trainer.predict(self.test_loader), f)

    def _make_metric_trackers(self):
        """Create new metric trackers for all metric functions"""
        train_metrics = {name:FnValueTracker(fn) for name, fn in self.train_metrics.items()}
        val_metrics = {name:FnValueTracker(fn) for name, fn in self.val_metrics.items()}
        return train_metrics, val_metrics

    def run(self, fold_num:int=None, use_checkpoint:bool=False):
        """
        Main method to run an experiment

        Parameters
        ----------
        fold_num
            optional, run only a certain fold

        use_checkpoint
            If True, will try to restore from a last checkpoint and continue to train untill max_iters is reached
        """
        for fold_i, (tr_ix, va_ix) in enumerate(self.splits):
            # start a new fold
            if (fold_num is not None) and (fold_i != fold_num):
                continue
            print("Started fold", fold_i)

            tr_loader, val_loader = self._make_loaders(tr_ix, va_ix)

            # instantiate Net model, optimizer, and maybe learning rate scheduler
            m = self.ModelClass(**self.model_params).to(self.device)
            opt = self.OptClass(m.parameters(), **self.opt_params)
            lr_sch = None if self.LrSchClass is None else self.LrSchClass(opt, **self.lr_sch_params)

            train_metrics, val_metrics = self._make_metric_trackers()

            # instantiate trainer object
            trainer = Trainer(m, self.loss_fn, opt, tr_loader, val_loader,
                              lr_schedule=lr_sch, lr_metric=self.lr_metric, pred_post_process=self.pred_post_process,
                              train_metrics=train_metrics, val_metrics=val_metrics,
                              max_iters=self.max_iters, device=self.device,
                              save_dir=os.path.join(self.experiment_dir, 'model'),
                              checkpoint_every=self.checkpoint_every,
                              name='model_{}'.format(fold_i))

            # maybe load latest checkpoint
            if use_checkpoint:
                try:
                    trainer.load_checkpoint()
                except:
                    print("failed to load a checkpoint, will start from scratch")

            # run training
            trainer.train()

            # save final
            trainer.save_checkpoint()

            # get final metrics
            self._add_metrics_to_meta(fold_i, trainer)

            # get predictions and save to work dir
            self._make_predictions(fold_i, trainer)

    def predict_from_checkpoints(self, fold_num:int=None):
        """
        Using latest checkpoint (re-)create predictions for whole training and test datasets
        Parameters
        ----------
        fold_num
            optional, only predict for a given fold number
        """
        for fold_i, _ in enumerate(self.splits):
            if (fold_num is not None) and (fold_i != fold_num):
                continue
            print("Loading model for fold", fold_i)

            # start model and optimizer (TODO: make optimizer optional)
            m = self.ModelClass(**self.model_params).to(self.device)
            opt = self.OptClass(m.parameters(), **self.opt_params)

            train_metrics, val_metrics = self._make_metric_trackers()

            # start trainer and load from latest checkpoint
            # no need to pass train and validation data loaders because whole datasets
            # loaders defined in __init__ will be used
            trainer = Trainer(m, self.loss_fn, opt, None, None, lr_schedule=None,
                              pred_post_process=self.pred_post_process,
                              train_metrics=train_metrics, val_metrics=val_metrics,
                              max_iters=self.max_iters, device=self.device,
                              save_dir=os.path.join(self.experiment_dir, 'model'),
                              name='model_{}'.format(fold_i))
            trainer.load_checkpoint()

            # update final metrics
            self._add_metrics_to_meta(fold_i, trainer)

            # get predictions and save to work dir
            self._make_predictions(fold_i, trainer)

    def make_submission(self, push=False, sample_file_path='sample_submission.csv',
                        mean_type='a', threshold=0.5, smooth=0.001, crop=None):

        test_df = pd.read_csv(sample_file_path, dtype=str).fillna('')

        all_preds = []
        for fold_i in tqdm(range(len(self.splits)), bar_format='reading fold predictions\t{l_bar}{bar}{r_bar}'):
            with open(os.path.join(self.experiment_dir, 'test_predictions_{}.pickle'.format(fold_i)), 'rb') as f:
                test_predictions = pickle.load(f)
                all_preds.append(test_predictions)

        predictions = np.concatenate([pred[None, ...] for pred in all_preds], axis=0)

        if mean_type == 'a':
            predictions = predictions.mean(axis=0)
        elif mean_type == 'g':
            predictions = np.exp(np.log(predictions + smooth).mean(axis=0))
        elif mean_type == 'max':
            predictions = predictions.max(axis=0)
        elif mean_type == 'min':
            predictions = predictions.min(axis=0)
        elif isinstance(mean_type, (int, float)):
            predictions = ((predictions + smooth) ** mean_type).mean(axis=0) ** (1 / mean_type)

        if crop is not None:
            predictions = predictions[:, :, 13:-14, 13:-14]

        predictions = predictions > threshold
        predictions = predictions[:, 0, :, :].transpose(0, 2, 1)

        n = predictions.shape[0]
        predictions = predictions.reshape((n, -1))

        masks_diffs = np.diff(np.concatenate([np.zeros((n, 1)), predictions, np.zeros((n, 1))], axis=1), axis=1)

        test_df['rle_mask'] = [to_rle(md)
                               for md in tqdm(masks_diffs, bar_format='making rle masks\t{l_bar}{bar}{r_bar}')]

        fpath = os.path.join(self.experiment_dir, 'subm_th-{}_mean-{}.csv'.format(threshold, mean_type))
        test_df.to_csv(fpath, index=False)
        print('saved submission to', fpath)

        # maybe load existing meta
        with open(os.path.join(self.experiment_dir, 'metadata.json')) as f:
            self.meta = json.load(f)

        self.meta['submissions'].append({
            'mean_type': mean_type,
            'threshold': threshold,
            'fpath': fpath,
            'score': None
        })
        # save to metadata
        with open(os.path.join(self.experiment_dir, 'metadata.json'), 'w') as f:
            json.dump(self.meta, f)


class Trainer(object):

    def __init__(self, model, loss_fn, optimizer, train_loader, val_loader, lr_schedule=None, lr_metric:str=None,
                 pred_post_process=None, train_metrics:dict=None, val_metrics:dict=None,
                 max_iters=10, checkpoint_every=50, device:str='cpu', save_dir:str='models', name:str=None):
        """
        Object responsible for training a neural netowrk model on given data

        need to produce
            - trained model
            - final train predictions
            - final test predictions
            - train loss and metric for all epochs
            - validation loss and metric for all epochs

        Parameters
        ----------
        model
            model to train

        loss_fn
            loss function

        optimizer
            optimizer

        train_loader
            training dataloader

        val_loader
            data loader with validation data, will be used to calculate at least loss after each
            training epoch, in addition metrics will be calculated if given

        lr_schedule
            optional, instance of learning rate scheduler

        lr_metric
            optional, name of the metric to be used with learning rate scheduler if needed,
            the metric must exist in the given `val_metrics`

        pred_post_process
            optional, function to call on predicted values, it will be called for each
            prediction, train, validation or test

        train_metrics
            optional, dict of {name:`FnValueTracker`}, metrics to be calculated for each batch of train data

        val_metrics
            optional, dict of {name:`FnValueTracker`}, validation metrics to be calculated for each epoch

        max_iters
            maximum number of iterations for training

        checkpoint_every
            how often to save checkpoints during training

        device
            pytorch device

        save_dir
            directory to be used for storing models

        name
            optional, will be used for naming directory and files of checkpoints,
            if not given, will be randomly generated
        """
        self.model_name = str(uuid.uuid4()) if name is None else name
        self.model = model
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.lr_metric = lr_metric
        self.pred_post_process = (lambda x: x) if pred_post_process is None else pred_post_process
        self.device = device

        self.save_dir = save_dir
        self.checkpoint_every = checkpoint_every
        self._last_checkpoint_fpath = None
        self.replace_checkpoint = True

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.max_iters = max_iters

        self.loss_fn = loss_fn

        self.train_metrics = {
            'loss': FnValueTracker(self.loss_fn)
        } if train_metrics is None else train_metrics

        self.val_metrics = {
            'loss': FnValueTracker(self.loss_fn)
        } if val_metrics is None else val_metrics

        self.curr_epoch = 0
        self._t_train_start = time.time()

        # helper for nice progress printing
        self.tqdm_default = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {elapsed}<{remaining}'

    def _make_tqdm_bar_for_train(self):
        """helper to print previous validation metrics in the progress bar"""
        suffix = ' Last Val:'
        if self.curr_epoch > 0:
            for n_name, tracker in self.val_metrics.items():
                suffix += "\t{:.3f}".format(tracker.get_last_epoch_mean())
        else:
            for n_name, tracker in self.val_metrics.items():
                suffix += f"\t{n_name}"

        elapsed_time = int(time.time() - self._t_train_start)
        time_str = '{:02}:{:02}'.format(elapsed_time//60, elapsed_time%60) if elapsed_time < 3600 \
            else '{}:{:02}:{:02}'.format(elapsed_time//3600, (elapsed_time%3600)//60, elapsed_time%60)

        return f'ep={self.curr_epoch} t={time_str}\t' + self.tqdm_default + suffix

    def _make_tqdm_bar_for_val(self):
        """helper to print previous train metrics in the progress bar"""
        suffix = ' Last Train:'
        if self.curr_epoch > 0:
            for n_name, tracker in self.train_metrics.items():
                suffix += "\t{:.3f}".format(tracker.get_last_epoch_mean())
        else:
            for n_name, tracker in self.train_metrics.items():
                suffix += f"\t{n_name}"

        elapsed_time = int(time.time() - self._t_train_start)
        time_str = '{:02}:{:02}'.format(elapsed_time // 60, elapsed_time % 60) if elapsed_time < 3600 \
            else '{}:{:02}:{:02}'.format(elapsed_time // 3600, (elapsed_time % 3600) // 60, elapsed_time % 60)

        return f'ep={self.curr_epoch} t={time_str}\t' + self.tqdm_default + suffix

    def add_metric(self, name, fn, train=True, val=True):
        """Add new metric functions to calculate and track"""
        if train:
            self.train_metrics[name] = FnValueTracker(fn)
        if val:
            self.val_metrics[name] = FnValueTracker(fn)

    def save_checkpoint(self):
        """
        Save a checkpoint,
        will save into {self.save_dir}/{self.model_name}/checkpoint_{epoch_num}

        File will have
            - net model state
            - optimizer state
            - metric trackers values
            - epoch number
        """
        os.makedirs(os.path.join(self.save_dir, self.model_name), exist_ok=True)
        fpath = os.path.join(self.save_dir, self.model_name, 'checkpoint_{}'.format(self.curr_epoch))

        self.model.to('cpu')

        # convert all components to cpu
        opt_state = self.optimizer.state_dict()
        for k in opt_state['state'].keys():
            opt_state['state'][k] = {sub_k: (v.cpu() if isinstance(v, torch.Tensor) else v) for sub_k, v in
                                     opt_state['state'][k].items()}

        torch.save({
            'model': self.model.state_dict(),
            'opt': self.optimizer.state_dict(),
            'train_metrics': {name: obj.values for name, obj in self.train_metrics.items()},
            'val_metrics': {name: obj.values for name, obj in self.val_metrics.items()},
            'curr_epoch': self.curr_epoch
        }, fpath)
        print("saved to", fpath)
        self.model.to(self.device)

        if self.replace_checkpoint and self._last_checkpoint_fpath is not None and self._last_checkpoint_fpath != fpath:
            os.unlink(self._last_checkpoint_fpath)
            print("removed", self._last_checkpoint_fpath)
        self._last_checkpoint_fpath = fpath

    def load_checkpoint(self, epoch=None):
        """
        Load from a checkpoint,
        If an epoch number is given, will try to load from {self.save_dir}/{self.model_name}/checkpoint_{epoch_num}
        Otherwise will find and load from a checkpoint with largest epoch number
        """
        if epoch is None:
            files = glob.glob(os.path.join(self.save_dir, self.model_name, 'checkpoint_*'))
            last_checkpoint = max([int(s.split("_")[-1]) for s in files])
        else:
            last_checkpoint = epoch
        fpath = os.path.join(self.save_dir, self.model_name, 'checkpoint_{}'.format(last_checkpoint))
        obj = torch.load(fpath)

        self.model.load_state_dict(obj['model'])
        self.model = self.model.to(self.device)

        for k in obj['opt']['state'].keys():
            obj['opt']['state'][k] = {sub_k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                                      for sub_k, v in obj['opt']['state'][k].items()}
        self.optimizer.load_state_dict(obj['opt'])

        self.curr_epoch = obj['curr_epoch']

        for name in self.train_metrics.keys():
            self.train_metrics[name].values = obj['train_metrics'][name]
        for name in self.val_metrics.keys():
            self.val_metrics[name].values = obj['val_metrics'][name]

        print("loaded from", fpath)

    def train(self):
        """Method to train a model"""
        print("starting to train")
        self._t_train_start = time.time()
        while self.curr_epoch < self.max_iters:
            self.train_epoch()
            self.validate_epoch()
            self.change_lr()
            self.curr_epoch += 1
            if (self.checkpoint_every is not None) and (self.curr_epoch % self.checkpoint_every == 0):
                self.save_checkpoint()
        print("finished training")

    def predict(self, data_loader, has_target=False):
        """
        Method to predict from input data given in the data_loader
        Parameters
        ----------
        data_loader
            data to use for prediction

        has_target
            If True, a batch is expected to have a target tensor that will be ignored
        Returns
        -------
            Numpy array with all predictions
        """
        self.model.eval()
        predictions = []
        tqdm_fmt = f'pred\t' + self.tqdm_default
        for i, batch in enumerate(tqdm(data_loader, bar_format=tqdm_fmt)):
            if has_target:
                x_b, _ = batch
            else:
                x_b = batch

            x_b = x_b.to(self.device)
            pred = self.model(x_b)
            pred = self.pred_post_process(pred)
            predictions.append(pred.detach().cpu().numpy())

        return np.concatenate(predictions, axis=0)

    def validate_epoch(self):
        """Helper to predict using evaluation data and calculate metrics"""
        self.model.eval()
        for batch_i, batch in enumerate(tqdm(self.val_loader, bar_format=self._make_tqdm_bar_for_val())):

            # get batch
            x_b, y_b = batch
            if isinstance(x_b, torch.Tensor):
                x_b = x_b.to(self.device)
            elif isinstance(x_b, (list, tuple)):
                x_b = [t.to(self.device) for t in x_b]

            if isinstance(y_b, torch.Tensor):
                y_b = y_b.to(self.device)
            elif isinstance(y_b, (list, tuple)):
                y_b = [t.to(self.device) for t in y_b]

            # predict
            pred = self.model(x_b)

            # post process
            pred = self.pred_post_process(pred)

            # calculate/save train losses
            for m_name, m_tracker in self.val_metrics.items():
                m_tracker(batch_i, self.curr_epoch, pred, y_b)

    def train_epoch(self):
        """Helper to run one training epoch"""
        self.model.train()
        for batch_i, batch in enumerate(tqdm(self.train_loader, bar_format=self._make_tqdm_bar_for_train())):

            # get batch
            x_b, y_b = batch
            if isinstance(x_b, torch.Tensor):
                x_b = x_b.to(self.device)
            elif isinstance(x_b, (list, tuple)):
                x_b = [t.to(self.device) for t in x_b]

            if isinstance(y_b, torch.Tensor):
                y_b = y_b.to(self.device)
            elif isinstance(y_b, (list, tuple)):
                y_b = [t.to(self.device) for t in y_b]

            # training step
            self.model.zero_grad()
            pred = self.model(x_b)
            loss = self.loss_fn(pred, y_b)
            loss.backward()
            self.optimizer.step()

            # post process
            pred = self.pred_post_process(pred)

            # calculate/save train losses
            self.train_metrics['loss'].add_value(batch_i, self.curr_epoch, loss.item())
            for m_name, m_tracker in self.train_metrics.items():
                if m_name != 'loss':
                    m_tracker(batch_i, self.curr_epoch, pred, y_b)
        self.model.eval()

    def change_lr(self):
        """Helper to change learning rate, if learning rate schedule is given"""
        if self.lr_schedule is not None:
            if self.lr_metric is not None:
                val = self.val_metrics[self.lr_metric].get_last_epoch_mean()
                self.lr_schedule.step(val)
            else:
                self.lr_schedule.step()

    def bag_net_inits(self):
        """Try several inits and select one with highest given metric on val set"""
        if self.curr_epoch == 0:
            print("trying out different inits")
            self.model.eval()
            results = []
            for try_i in range(10):
                self.model.reset_parameters()
                self.validate_epoch()
                self.model.to('cpu')
                results.append({
                    'metric': self.val_metrics[self.lr_metric].get_last_epoch_mean(),
                    'model_state': self.model.state_dict()
                })
                self.val_metrics[self.lr_metric].reset()
                self.model.to(self.device)
            best = np.argmax([r['metric'] for r in results])
            self.model.to('cpu')
            self.model.load_state_dict(results[best]['model_state'])
            self.model.to(self.device)