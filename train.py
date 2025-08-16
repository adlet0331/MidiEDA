from datetime import datetime
import numpy as np
import os

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
from torch.utils.tensorboard import SummaryWriter

from model import *

@ex.config
def my_config():
    # runs/p-est-250309-211211
    logdir = 'runs/p-est-' + datetime.now().strftime('%y%m%d-%H%M%S')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iterations = 10000
    batch_size = 32
    resume_iteration = None

    learning_rate = 1e-2
    learning_rate_decay_steps = 1000
    learning_rate_decay_rate = 0.9

    accumulate_steps = 2  # Gradient Accumulation Steps
    validation_interval = 100 * accumulate_steps
    checkpoint_interval = 1000 * accumulate_steps

    numeric_versions = 1 # Numeric Features 버전, 버전별로 metadata에 저장
    numeric_features = 14  # Mikrokosmos의 Numeric Features 수

    seed = 42
    save_log = False  # 로그 저장 여부
    ex.observers.append(FileStorageObserver(logdir))

@ex.automain
def train(logdir, numeric_features, numeric_versions, device, iterations, batch_size, resume_iteration, 
          learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, 
          validation_interval, accumulate_steps, checkpoint_interval, save_log, _seed):
    print_config(ex.current_run)

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    torch.manual_seed(_seed)
    np.random.seed(_seed)

    # 데이터셋 로드
    mikrokosmos_dataset = MikrokosmosDataset()
    cipi_dataset = CipiDataset()

    combined_dataset = ConcatDataset([mikrokosmos_dataset, cipi_dataset])
    trainset, validset = random_split(
        combined_dataset,
        [int(len(combined_dataset) * 0.8),
         len(combined_dataset) - int(len(combined_dataset) * 0.8)]
    )

    # DataLoader 설정
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(validset, batch_size=batch_size)

    model = RubricNet(num_features=numeric_features).to(device, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate
    )

    # WanDB Logging 예시
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
                "resume_iteration": resume_iteration,
                "iterations": iterations,
                "accumulate_steps": accumulate_steps,
                "validation_interval": validation_interval,
                "checkpoint_interval": checkpoint_interval,
                "device": device,
                "seed": _seed
            }
        )
    
    # 모델 학습 재개시
    if resume_iteration is not None:
        model.load_state_dict(torch.load(
            os.path.join(logdir, 'model_snapshots', f'model_{resume_iteration//checkpoint_interval}.pt')
        ))
    else:
        os.makedirs(os.path.join(logdir, 'model_snapshots'), exist_ok=True)

    print(f"학습을 시작합니다. 총 {iterations}번의 반복을 수행합니다.")

    best_model_path = None
    best_validation_loss = float('inf')

    # model = torch.compile(model)  # PyTorch 2.0 이상에서 사용 가능
    model.train()
    loss_function = BCEWithLogitsLoss()
    
    # ====== 학습 루프 (최소 수정: zip 제거, while+for로 반복 유지) ======
    iteration = (resume_iteration or 0)
    while iteration < iterations:
        for batch_features, batch_labels in train_dataloader:
            iteration += 1

            # 학습 코드
            batch_features = batch_features.to(device, dtype=torch.float32)
            batch_labels = (torch.arange(1, 10, device=device)[None, :] <= batch_labels[:, None]).to(torch.float32)
            predictions = model(batch_features)

            train_loss = loss_function(predictions, batch_labels)
            train_loss.backward()

            print(f"Iteration {iteration}, Train Loss: {train_loss.item():.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            if save_log:
                wandb.log({"train_loss": train_loss.item()}, step=iteration)
                writer.add_scalar('train_loss', train_loss.item(), global_step=iteration)

            if iteration % accumulate_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if iteration % validation_interval == 0:
                model.eval()
                validation_loss = 0.0
                with torch.no_grad():
                    for valid_batch in valid_dataloader:
                        valid_batch_features, valid_batch_labels = valid_batch
                        valid_batch_features = valid_batch_features.to(device, dtype=torch.float32)
                        valid_batch_labels = (
                            torch.arange(1, 10, device=device)[None, :] <= valid_batch_labels[:, None]
                        ).to(torch.float32)
                        valid_predictions = model(valid_batch_features)
                        validation_loss += loss_function(valid_predictions, valid_batch_labels).sum()
                validation_loss /= len(valid_dataloader)

                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_model_path = os.path.join(logdir, 'model_snapshots', f'model_bestvalidation.pt')
                    torch.save(model.state_dict(), best_model_path)
                    print(f"Best model saved at iteration {iteration} with validation loss {best_validation_loss:.4f}")

                if save_log:
                    wandb.log({"validation_loss": validation_loss}, step=iteration)
                    writer.add_scalar('validation_loss', validation_loss, global_step=iteration)

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