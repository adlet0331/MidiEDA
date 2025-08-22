from datetime import datetime
import numpy as np
import os
import json  # JSON 저장을 위해 추가

from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from sacred import SETTINGS
SETTINGS.CAPTURE_MODE = 'sys' 

ex = Experiment('my_experiment')

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split, ConcatDataset, DataLoader

from sklearn.metrics import balanced_accuracy_score
from statistics import mean, stdev

from model import *

@ex.config
def my_config():
    # runs/p-est-250309-211211
    logdir = 'runs/p-est-' + datetime.now().strftime('%y%m%d-%H%M%S')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iterations = 100000
    batch_size = 32

    learning_rate = 1e-2
    learning_rate_decay_steps = 1000
    learning_rate_decay_rate = 0.9

    accumulate_steps = 2  # Gradient, Optimizer, Scheduler Accumulation Steps
    validation_interval = 200 // accumulate_steps
    checkpoint_interval = 1000

    numeric_versions = 1 # Numeric Features 버전, 버전별로 metadata에 저장
    numeric_features = 14  # Mikrokosmos의 Numeric Features 수

    seed = 42
    save_log = True  # 로그 저장 여부
    ex.observers.append(FileStorageObserver(logdir))

@ex.automain
def train(logdir, numeric_features, numeric_versions, device, iterations, batch_size, 
          learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, 
          validation_interval, accumulate_steps, checkpoint_interval, save_log, _seed):
    print_config(ex.current_run)

    if save_log:
        os.makedirs(logdir, exist_ok=True)
    torch.manual_seed(_seed)
    np.random.seed(_seed)

    # 데이터셋 로드
    mikrokosmos_dataset = MikrokosmosDataset(dataset_path=MIKROKOSMOS_PATH, numeric_versions=numeric_versions)
    cipi_dataset = CipiDataset(dataset_path=CIPI_PATH, numeric_versions=numeric_versions)

    combined_dataset = ConcatDataset([mikrokosmos_dataset, cipi_dataset])
    trainset, validset, testset = random_split(
        combined_dataset,
        [int(len(combined_dataset) * 0.6),
         int(len(combined_dataset) * 0.2),
         len(combined_dataset) - int(len(combined_dataset) * 0.6) - int(len(combined_dataset) * 0.2)]
    )

    # ==================== [추가된 코드 시작] ====================
    # train/validation set의 인덱스를 json 파일로 저장
    if save_log:
        dataset_split_info = {
            'train': trainset.indices,
            'validation': validset.indices,
            'test': testset.indices
        }
        split_file_path = os.path.join(logdir, 'dataset_split.json')
        with open(split_file_path, 'w') as f:
            json.dump(dataset_split_info, f, indent=4)
        print(f"Train/validation/test split indices saved to {split_file_path}")
    # ==================== [추가된 코드 끝] ====================

    # DataLoader 설정
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(validset, batch_size=batch_size)
    test_dataloader = DataLoader(testset, batch_size=batch_size)

    features_list = []
    features_name_list = mikrokosmos_dataset.features_names
    for batch_features, _ in train_dataloader:
        features_list.append(batch_features.detach().cpu().numpy())
    features_list = np.concatenate(features_list, axis=0)

    feature_mean_std_list = [[], []]
    for i in range(numeric_features):
        feature_list = features_list[:, i].reshape(-1, 1)
        feature_mean_std_list[0].append(np.mean(feature_list))
        feature_mean_std_list[1].append(np.std(feature_list))
        print(f"{features_name_list[i]} scaler: {feature_mean_std_list[0][i]}, {feature_mean_std_list[1][i]}")

    model = RubricNet(
        num_features=numeric_features, 
        scaler_parameter=feature_mean_std_list,
        performance_top=9,
        threshold=0.5,
        dropout_rate=0.2
    ).to(device, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate
    )

    if save_log:
        import wandb
        wandb.init(
            project="RubricNet",
            name=logdir,
            config={
                "logdir": logdir,
                "learning_rate": learning_rate,
                "learning_rate_decay_steps": learning_rate_decay_steps,
                "batch_size": batch_size,
                "iterations": iterations,
                "accumulate_steps": accumulate_steps,
                "validation_interval": validation_interval,
                "checkpoint_interval": checkpoint_interval,
                "device": device,
                "seed": _seed
            }
        )
    
    os.makedirs(os.path.join(logdir, 'model_snapshots'), exist_ok=True)
    print(f"학습을 시작합니다. 총 {iterations}번의 반복을 수행합니다.")

    iteration = 0
    best_model_path = None
    best_validation_loss = float('inf')

    # model = torch.compile(model)  # PyTorch 2.0 이상에서 사용 가능
    model.train()
    loss_function = BCEWithLogitsLoss()

    # ====== 학습 루프 시작 ======
    while iteration < iterations:
        for batch_features, batch_labels in train_dataloader:
            iteration += 1

            # 학습 코드
            batch_features = batch_features.to(device, dtype=torch.float32)
            batch_labels = (torch.arange(1, 10, device=device)[None, :] <= batch_labels[:, None]).to(torch.float32)
            predictions = model(batch_features)

            train_loss = loss_function(predictions, batch_labels)
            train_loss.backward()

            if save_log:
                wandb.log({"train_loss": train_loss.item()}, step=iteration)
                ex.log_scalar('train_loss', train_loss.item(), iteration)

            if iteration % accumulate_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if iteration % validation_interval == 0:
                print(f"[[Iteration {iteration}]] Train Loss: {train_loss.item():.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
                mse_scores = []
                acc1_scores = []
                acc3_scores = []
                acc9_scores = []
                validation_loss = 0.0
                
                # Validation 코드
                with torch.no_grad():
                    model.eval()
                    for valid_batch in valid_dataloader:
                        valid_batch_features, valid_batch_labels = valid_batch
                        valid_batch_features = valid_batch_features.to(device, dtype=torch.float32)
                        valid_batch_labels = (
                            torch.arange(1, 10, device=device)[None, :] <= valid_batch_labels[:, None]
                        ).to(torch.float32)
                        valid_predictions = model(valid_batch_features)
                        inference = inference_from_pred(valid_predictions)
                        valid_batch_labels = inference_from_pred(valid_batch_labels)

                        acc1_scores.append(get_acc1_macro(y_true=valid_batch_labels, y_pred=inference))
                        mse_scores.append(get_mse_macro(y_true=valid_batch_labels, y_pred=inference))

                        acc9_scores.append(balanced_accuracy_score(y_true=valid_batch_labels, y_pred=inference))
                        nine2three = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
                        acc3_scores.append(balanced_accuracy_score(y_true=[nine2three[x] for x in valid_batch_labels],
                                                                    y_pred=[nine2three[x] for x in inference]))

                if mse_scores:
                    validation_loss = sum(mse_scores) / len(mse_scores)

                print(f"""\t Validation Loss: {validation_loss:.4f}
\t MSE Score: {mean(mse_scores):.4f}
\t Acc1 Score: {mean(acc1_scores):.4f}
\t Acc3 Score: {mean(acc3_scores):.4f}
\t Acc9 Score: {mean(acc9_scores):.4f}""")

                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_model_path = os.path.join(logdir, 'model_snapshots', f'model_bestvalidation.pt')
                    torch.save(model.state_dict(), best_model_path)
                    print(f"Best model saved at iteration {iteration} with validation loss {best_validation_loss:.4f}")

                if save_log:
                    wandb.log({
                        "validation_loss": validation_loss,
                        "val_mse_score": mean(mse_scores),
                        "val_acc1_score": mean(acc1_scores),
                        "val_acc3_score": mean(acc3_scores),
                        "val_acc9_score": mean(acc9_scores)
                    }, step=iteration)
                    ex.log_scalar('validation_loss', validation_loss, iteration)
                    ex.log_scalar('val_mse_score', mean(mse_scores), iteration)
                    ex.log_scalar('val_acc1_score', mean(acc1_scores), iteration)
                    ex.log_scalar('val_acc3_score', mean(acc3_scores), iteration)
                    ex.log_scalar('val_acc9_score', mean(acc9_scores), iteration)

                # Test 결과 저장
                mse_scores = []
                acc1_scores = []
                acc3_scores = []
                acc9_scores = []
                test_loss = 0.0

                # Test 코드
                with torch.no_grad():
                    model.eval()
                    for test_batch in test_dataloader:
                        test_batch_features, test_batch_labels = test_batch
                        test_batch_features = test_batch_features.to(device, dtype=torch.float32)
                        test_batch_labels = (
                            torch.arange(1, 10, device=device)[None, :] <= test_batch_labels[:, None]
                        ).to(torch.float32)
                        test_predictions = model(test_batch_features)
                        inference = inference_from_pred(test_predictions)
                        test_batch_labels = inference_from_pred(test_batch_labels)

                        acc1_scores.append(get_acc1_macro(y_true=test_batch_labels, y_pred=inference))
                        mse_scores.append(get_mse_macro(y_true=test_batch_labels, y_pred=inference))

                        acc9_scores.append(balanced_accuracy_score(y_true=test_batch_labels, y_pred=inference))
                        nine2three = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
                        acc3_scores.append(balanced_accuracy_score(y_true=[nine2three[x] for x in test_batch_labels],
                                                                    y_pred=[nine2three[x] for x in inference]))

                if mse_scores:
                    test_loss = sum(mse_scores) / len(mse_scores)

                print(f"""\t Test Loss: {test_loss:.4f}
\t MSE Score: {mean(mse_scores):.4f}
\t Acc1 Score: {mean(acc1_scores):.4f}
\t Acc3 Score: {mean(acc3_scores):.4f}
\t Acc9 Score: {mean(acc9_scores):.4f}""")

                if save_log:
                    wandb.log({
                        "test_loss": test_loss,
                        "test_mse_score": mean(mse_scores),
                        "test_acc1_score": mean(acc1_scores),
                        "test_acc3_score": mean(acc3_scores),
                        "test_acc9_score": mean(acc9_scores)
                    }, step=iteration)
                    ex.log_scalar('test_loss', test_loss, iteration)
                    ex.log_scalar('test_mse_score', mean(mse_scores), iteration)
                    ex.log_scalar('test_acc1_score', mean(acc1_scores), iteration)
                    ex.log_scalar('test_acc3_score', mean(acc3_scores), iteration)
                    ex.log_scalar('test_acc9_score', mean(acc9_scores), iteration)

                model.train()

            if iteration % checkpoint_interval == 0:
                checkpoint_path = os.path.join(logdir, 'model_snapshots', f'model_{iteration // checkpoint_interval}.pt')
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved at iteration {iteration} to {checkpoint_path}")

            if iteration >= iterations:
                break
    # ====== 학습 루프 끝 ======
    print(f"학습이 완료되었습니다. 총 {iteration}번의 반복을 수행했습니다.")
    print(f"Best validation loss: {best_validation_loss:.4f}")
    print(f"저장 경로: logdir: {logdir}")