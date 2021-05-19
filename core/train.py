#!/usr/bin/env python3
"""
Script for training CG-GNN, TG-GNN and HACT models
"""
import torch
import mlflow
import os
import uuid
import yaml
from tqdm import tqdm
import mlflow.pytorch
import numpy as np
import pandas as pd
import shutil
import argparse
from sklearn.metrics import accuracy_score, f1_score, classification_report

from histocartography.ml import CellGraphModel, TissueGraphModel, HACTModel

from dataloader import make_data_loader

# cuda support
IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'
NODE_DIM = 514


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cg_path',
        type=str,
        help='path to the cell graphs.',
        default=None,
        required=False
    )
    parser.add_argument(
        '--tg_path',
        type=str,
        help='path to tissue graphs.',
        default=None,
        required=False
    )
    parser.add_argument(
        '--assign_mat_path',
        type=str,
        help='path to the assignment matrices.',
        default=None,
        required=False
    )
    parser.add_argument(
        '-conf',
        '--config_fpath',
        type=str,
        help='path to the config file.',
        default='',
        required=False
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help='path to where the model is saved.',
        default='',
        required=False
    )
    parser.add_argument(
        '--in_ram',
        help='if the data should be stored in RAM.',
        action='store_true',
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        help='batch size.',
        default=1,
        required=False
    )
    parser.add_argument(
        '--epochs', type=int, help='epochs.', default=10, required=False
    )
    parser.add_argument(
        '-l',
        '--learning_rate',
        type=float,
        help='learning rate.',
        default=10e-3,
        required=False
    )
    parser.add_argument(
        '--out_path',
        type=str,
        help='path to where the output data are saved (currently only for the interpretability).',
        default='../../data/graphs',
        required=False
    )
    parser.add_argument(
        '--logger',
        type=str,
        help='Logger type. Options are "mlflow" or "none"',
        required=False,
        default='none'
    )

    return parser.parse_args()

def main(args):
    """
    Train HACTNet, CG-GNN or TG-GNN.
    Args:
        args (Namespace): parsed arguments.
    """

    # load config file
    with open(args.config_fpath, 'r') as f:
        config = yaml.load(f)

    # log parameters to logger
    if args.logger == 'mlflow':
        mlflow.log_params({
            'batch_size': args.batch_size
        })
        df = pd.io.json.json_normalize(config)
        rep = {"graph_building.": "", "model_params.": "", "gnn_params.": ""}  # replacement for shorter key names
        for i, j in rep.items():
            df.columns = df.columns.str.replace(i, j)
        flatten_config = df.to_dict(orient='records')[0]
        for key, val in flatten_config.items():
            mlflow.log_params({key: str(val)})

    # set path to save checkpoints 
    model_path = os.path.join(args.model_path, str(uuid.uuid4()))
    os.makedirs(model_path, exist_ok=True)

    # make data loaders (train, validation & test)
    train_dataloader = make_data_loader(
        cg_path=os.path.join(args.cg_path, 'train') if args.cg_path is not None else None,
        tg_path=os.path.join(args.tg_path, 'train') if args.tg_path is not None else None,
        assign_mat_path=os.path.join(args.assign_mat_path, 'train') if args.assign_mat_path is not None else None,
        batch_size=args.batch_size,
        load_in_ram=args.in_ram,
    )
    val_dataloader = make_data_loader(
        cg_path=os.path.join(args.cg_path, 'val') if args.cg_path is not None else None,
        tg_path=os.path.join(args.tg_path, 'val') if args.tg_path is not None else None,
        assign_mat_path=os.path.join(args.assign_mat_path, 'val') if args.assign_mat_path is not None else None,
        batch_size=args.batch_size,
        load_in_ram=args.in_ram,
    )
    test_dataloader = make_data_loader(
        cg_path=os.path.join(args.cg_path, 'test') if args.cg_path is not None else None,
        tg_path=os.path.join(args.tg_path, 'test') if args.tg_path is not None else None,
        assign_mat_path=os.path.join(args.assign_mat_path, 'test') if args.assign_mat_path is not None else None,
        batch_size=args.batch_size,
        load_in_ram=args.in_ram,
    )

    # declare model 
    if 'bracs_cggnn' in args.config_fpath:
        model = CellGraphModel(
            gnn_params=config['gnn_params'],
            classification_params=config['classification_params'],
            node_dim=NODE_DIM,
            num_classes=7
        ).to(DEVICE)

    elif 'bracs_tggnn' in args.config_fpath:
        model = TissueGraphModel(
            gnn_params=config['gnn_params'],
            classification_params=config['classification_params'],
            node_dim=NODE_DIM,
            num_classes=7
        ).to(DEVICE)

    elif 'bracs_hact' in args.config_fpath:
        model = HACTModel(
            cg_gnn_params=config['cg_gnn_params'],
            tg_gnn_params=config['tg_gnn_params'],
            classification_params=config['classification_params'],
            cg_node_dim=NODE_DIM,
            tg_node_dim=NODE_DIM,
            num_classes=7
        ).to(DEVICE)
    else:
        raise ValueError('Model type not recognized. Options are: TG, CG or HACT.')

    # build optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=5e-4
    )

    # define loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # training loop
    step = 0
    best_val_loss = 10e5
    best_val_accuracy = 0.
    best_val_weighted_f1_score = 0.

    for epoch in range(args.epochs):
        # A.) train for 1 epoch
        model = model.to(DEVICE)
        model.train()
        for batch in tqdm(train_dataloader, desc='Epoch training {}'.format(epoch), unit='batch'):

            # 1. forward pass
            labels = batch[-1]
            data = batch[:-1]
            logits = model(*data)

            # 2. backward pass
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 3. log training loss 
            if args.logger == 'mlflow':
                mlflow.log_metric('train_loss', loss.item(), step=step)

            # 4. increment step
            step += 1

        # B.) validate
        model.eval()
        all_val_logits = []
        all_val_labels = []
        for batch in tqdm(val_dataloader, desc='Epoch validation {}'.format(epoch), unit='batch'):
            labels = batch[-1]
            data = batch[:-1]
            with torch.no_grad():
                logits = model(*data)
            all_val_logits.append(logits)
            all_val_labels.append(labels)

        all_val_logits = torch.cat(all_val_logits).cpu()
        all_val_preds = torch.argmax(all_val_logits, dim=1)
        all_val_labels = torch.cat(all_val_labels).cpu()

        # compute & store loss + model
        with torch.no_grad():
            loss = loss_fn(all_val_logits, all_val_labels).item()
        if args.logger == 'mlflow':
            mlflow.log_metric('val_loss', loss, step=step)
        if loss < best_val_loss:
            best_val_loss = loss
            torch.save(model.state_dict(), os.path.join(model_path, 'model_best_val_loss.pt'))

        # compute & store accuracy + model
        all_val_preds = all_val_preds.detach().numpy()
        all_val_labels = all_val_labels.detach().numpy()
        accuracy = accuracy_score(all_val_labels, all_val_preds)
        if args.logger == 'mlflow':
            mlflow.log_metric('val_accuracy', accuracy, step=step)
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(model_path, 'model_best_val_accuracy.pt'))

        # compute & store weighted f1-score + model
        weighted_f1_score = f1_score(all_val_labels, all_val_preds, average='weighted')
        if args.logger == 'mlflow':
            mlflow.log_metric('val_weighted_f1_score', weighted_f1_score, step=step)
        if weighted_f1_score > best_val_weighted_f1_score:
            best_val_weighted_f1_score = weighted_f1_score
            torch.save(model.state_dict(), os.path.join(model_path, 'model_best_val_weighted_f1_score.pt'))

        print('Val loss {}'.format(loss))
        print('Val weighted F1 score {}'.format(weighted_f1_score))
        print('Val accuracy {}'.format(accuracy))

    # testing loop
    model.eval()
    for metric in ['best_val_loss', 'best_val_accuracy', 'best_val_weighted_f1_score']:
        
        print('\n*** Start testing w/ {} model ***'.format(metric))

        model_name = [f for f in os.listdir(model_path) if f.endswith(".pt") and metric in f][0]
        model.load_state_dict(torch.load(os.path.join(model_path, model_name)))

        all_test_logits = []
        all_test_labels = []
        for batch in tqdm(test_dataloader, desc='Testing: {}'.format(metric), unit='batch'):
            labels = batch[-1]
            data = batch[:-1]
            with torch.no_grad():
                logits = model(*data)
            all_test_logits.append(logits)
            all_test_labels.append(labels)

        all_test_logits = torch.cat(all_test_logits).cpu()
        all_test_preds = torch.argmax(all_test_logits, dim=1)
        all_test_labels = torch.cat(all_test_labels).cpu()

        # compute & store loss
        with torch.no_grad():
            loss = loss_fn(all_test_logits, all_test_labels).item()
        if args.logger == 'mlflow':
            mlflow.log_metric('best_test_loss_' + metric, loss)

        # compute & store accuracy
        all_test_preds = all_test_preds.detach().numpy()
        all_test_labels = all_test_labels.detach().numpy()
        accuracy = accuracy_score(all_test_labels, all_test_preds)
        if args.logger == 'mlflow':
            mlflow.log_metric('best_test_accuracy_' + metric, accuracy, step=step)

        # compute & store weighted f1-score
        weighted_f1_score = f1_score(all_test_labels, all_test_preds, average='weighted')
        if args.logger == 'mlflow':
            mlflow.log_metric('best_test_weighted_f1_score_' + metric, weighted_f1_score, step=step)

        # compute and store classification report 
        report = classification_report(all_test_labels, all_test_preds)
        out_path = os.path.join(model_path, 'classification_report.txt')
        with open(out_path, "w") as f:
            f.write(report)

        if args.logger == 'mlflow':
            artifact_path = 'evaluators/class_report_{}'.format(metric)
            mlflow.log_artifact(out_path, artifact_path=artifact_path)

        # log MLflow models
        mlflow.pytorch.log_model(model, 'model_' + metric)

        print('Test loss {}'.format(loss))
        print('Test weighted F1 score {}'.format(weighted_f1_score))
        print('Test accuracy {}'.format(accuracy))


    if args.logger == 'mlflow':
        shutil.rmtree(model_path)


if __name__ == "__main__":
    main(args=parse_arguments())
