import argparse,json, os, sys, time

import numpy as np 
import torch 
import torch.nn as nn
import torch.optim as optim

import feater
from feater import utils
from feater.models import fused


def test_model(model, data1, data2, criterion, batch_size=128, worker_nr=12):
  model = model.eval()
  with torch.no_grad(): 
    batch_to_test = np.arange(len(data1))
    np.random.shuffle(batch_to_test)
    batches = feater.dataloader.split_array(batch_to_test, batch_size)
    accuracy_test = 0
    loss_test = 0 
    for bidx, b in enumerate(batches):
      d1_test, d1_label_test = data1.get_batch_by_index(b, worker_nr)
      d2_test, d2_label_test = data2.get_batch_by_index(b, worker_nr)
      d1_test, d1_label_test = d1_test.cuda(), d1_label_test.cuda()
      d2_test, d2_label_test = d2_test.cuda(), d2_label_test.cuda()

      pred_test = model(d1_test, d2_test)
      accuracy_test += (torch.argmax(pred_test, dim=1) == d1_label_test).sum().item()
      loss_test += criterion(pred_test, d1_label_test).item()
    
    accuracy_test /= len(data1)
    loss_test /= len(data1)
  return accuracy_test, loss_test



def parser(): 
  parser = argparse.ArgumentParser(description="Train a fused model")

  parser.add_argument("-m", "--model", type=str, required=True, default="gnina_pointnet", help="Model to train")
  parser.add_argument("--optimizer", type=str, default="adam", help="The optimizer to use")
  parser.add_argument("--loss-function", type=str, default="crossentropy", help="The loss function to use")

  # Data files 
  parser.add_argument("-train", "--training-data", type=str, required=True, help="The file writes all of the absolute path of h5 files of training data set")
  parser.add_argument("-test", "--test-data", type=str, required=True, help="The file writes all of the absolute path of h5 files of test data set")
  parser.add_argument("-o", "--output-folder", type=str, required=True, help="The output folder to store the model and performance data")
  parser.add_argument("-w", "--data-workers", type=int, default=12, help="Number of workers for data loading")
  parser.add_argument("--test-number", type=int, default=4000, help="Number of test samples to use")


  # Pretrained model and break point restart
  parser.add_argument("--pretrained", type=str, default=None, help="Pretrained model path")
  parser.add_argument("--start-epoch", type=int, default=0, help="Start epoch")

  # Training parameters
  parser.add_argument("-b", "--batch-size", type=int, default=128, help="Batch size")
  parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs")
  parser.add_argument("-lr", "--lr-init", type=float, default=0.001, help="Learning rate")
  parser.add_argument("--lr-decay-steps", type=int, default=30, help="Decay the learning rate every n steps")
  parser.add_argument("--lr-decay-rate", type=float, default=0.5, help="Decay the learning rate by this rate")

  # Miscellanous arguments 
  parser.add_argument("-s", "--manualSeed", type=int, help="Manually set seed")

  # PointNet specific arguments 
  parser.add_argument("--target_np", type=int, default=24, help="Target number of points for PointNet")
  parser.add_argument("--target_np2", type=int, default=1500, help="Target number of points for PointNet if the model is fused with PointNet")

  args = parser.parse_args()
  return args

def perform_training(settings, model, data1, data2, data1_test, data2_test):

  EPOCH_NR = settings["epochs"]
  BATCH_SIZE = settings["batch_size"]
  START_EPOCH = settings["start_epoch"]
  WORKER_NR = settings["data_workers"]
  MODEL_TYPE = settings["model"]
  LR_INIT = settings["lr_init"]

  optimizer = optim.Adam(model.parameters(), lr=LR_INIT, betas=(0.9, 0.999))
  criterion = torch.nn.CrossEntropyLoss()
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=settings["lr_decay_steps"], gamma=settings["lr_decay_rate"])
  
  for epoch in range(0, EPOCH_NR): 
    st = time.perf_counter()
    st_epoch = time.perf_counter()
    if (epoch < START_EPOCH): 
      print(f"Skip the epoch {epoch}/{START_EPOCH} ...")
      scheduler.step()
      print(f"Epoch {epoch} took {time.perf_counter() - st_epoch:6.2f} seconds to train. Current learning rate: {feater.utils.get_lr(optimizer):.6f}. ")
      continue
    message = f" Running the epoch {epoch:>4d}/{EPOCH_NR:<4d} at {time.ctime()}; learning rate {feater.utils.get_lr(optimizer):.6f} "
    print(f"{message:#^80}")
    
    # Generate different batches for each epoch 
    arr = np.arange(len(data1))
    np.random.shuffle(arr)
    batches = feater.dataloader.split_array(arr, BATCH_SIZE)
    for bidx, b in enumerate(batches): 
      d1, d1_label = data1.get_batch_by_index(b, WORKER_NR)
      d2, d2_label = data2.get_batch_by_index(b, WORKER_NR)
      if False in (d1_label == d2_label): 
        raise ValueError("Label mismatch")
      if len(d1_label) != BATCH_SIZE: 
        print(f"The final batch size is {len(d1_label)}, skip the batch ...")
        continue  

      d1, d1_label = d1.cuda(), d1_label.cuda()
      d2, d2_label = d2.cuda(), d2_label.cuda()
      
      optimizer.zero_grad()
      model = model.train()
      pred = model(d1, d2)
      loss = criterion(pred, d1_label)
      loss.backward()
      optimizer.step()
      accuracy = (torch.argmax(pred, dim=1) == d1_label).sum().item() / len(d1_label)
      
      if (bidx+1) % (len(batches) // 10) == 0: 
        # for name, param in model.model1.named_parameters(): 
        #   print(name, param.requires_grad, None if param.grad is None else param.grad.norm())
        model = model.eval()
        with torch.no_grad(): 
          batch_to_test = np.arange(len(data1_test))
          np.random.shuffle(batch_to_test)
          d1_test, d1_label_test = data1_test.get_batch_by_index(batch_to_test[:128], WORKER_NR)
          d2_test, d2_label_test = data2_test.get_batch_by_index(batch_to_test[:128], WORKER_NR)
          d1_test, d1_label_test = d1_test.cuda(), d1_label_test.cuda()
          d2_test, d2_label_test = d2_test.cuda(), d2_label_test.cuda()

          ################################################
          if perform_recording:
            parmset = list(model.named_parameters())
            for idx, p in enumerate(parmset):
              pname = p[0]
              p = p[1]
              if p.grad is not None:
                gradi = p.grad.cpu().detach().numpy()
                tensorboard_writer.add_scalar(f"norm/layer{idx}_{pname}", np.linalg.norm(gradi), epoch * len(batches) + bidx)
                tensorboard_writer.add_histogram(f"hist/layer{idx}_{pname}", gradi, epoch * len(batches) + bidx)
          ################################################
          
          pred_test = model(d1_test, d2_test)
          accuracy_test = (torch.argmax(pred_test, dim=1) == d1_label_test).sum().item() / len(d1_label_test)
          print(f"Accuracy on the Test set: {accuracy_test:5.3f}/{accuracy:5.3f}")
    
    # Save the model  
    modelfile_output = os.path.join(os.path.abspath(settings["output_folder"]), f"{MODEL_TYPE}_{epoch}.pth")
    print(f"Saving the model to {modelfile_output} ...")
    torch.save(model.state_dict(), modelfile_output)
    scheduler.step()
    
    acc_i, loss_i = test_model(model, data1_test, data2_test, criterion, batch_size=BATCH_SIZE, worker_nr=WORKER_NR)
    print(f"Epoch {epoch} took {time.perf_counter() - st_epoch:6.2f} seconds to train, Accuracy {acc_i}. Current learning rate: {feater.utils.get_lr(optimizer):.6f}. ")

    with feater.io.hdffile(os.path.join(settings["output_folder"], "performance.h5") , "a") as hdffile:
      utils.update_hdf_by_slice(hdffile, "loss_train", np.array([loss.item()], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None, )) 
      utils.update_hdf_by_slice(hdffile, "loss_test", np.array([loss_i], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None, ))
      utils.update_hdf_by_slice(hdffile, "accuracy_train", np.array([accuracy], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None, ))
      utils.update_hdf_by_slice(hdffile, "accuracy_test", np.array([acc_i], dtype=np.float64), np.s_[epoch:epoch+1], dtype=np.float64, maxshape=(None, ))
  


if __name__ == "__main__":
  perform_recording = 0 
  if perform_recording: 
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_writer = SummaryWriter(os.path.join("/tmp", "tensorboard"))

  args = parser()
  settings = vars(args)
  print(json.dumps(settings, indent=2))

  raw_tr_files = settings["training_data"].strip().strip("%").split("%")
  raw_te_files = settings["test_data"].strip().strip("%").split("%")

  if len(raw_tr_files) != 2: 
    raise ValueError("Training data files are not provided correctly")
  if len(raw_te_files) != 2:
    raise ValueError("Test data files are not provided correctly")
  
  datafile1 = [raw_tr_files[0]]
  datafile2 = [raw_tr_files[1]]
  testdata1 = [raw_te_files[0]]
  testdata2 = [raw_te_files[1]]

  if "single" in settings["output_folder"]: 
    output_dim = 20
  elif "dual" in settings["output_folder"]:
    output_dim = 400
  else: 
    raise ValueError("Output dimension not identified")
  
  if "gnina_pointnet" in settings["model"]:
    model = fused.gnina_pointnet(1, output_dim, attention=False)
    data1 = feater.dataloader.VoxelDataset(datafile1)
    data1_test = feater.dataloader.VoxelDataset(testdata1)

    if settings["model"].endswith("_s"): 
      data2 = feater.dataloader.SurfDataset(datafile2, target_np=settings["target_np2"])
      data2_test = feater.dataloader.SurfDataset(testdata2, target_np=settings["target_np2"])
    else:
      data2 = feater.dataloader.CoordDataset(datafile2, target_np=settings["target_np"])
      data2_test = feater.dataloader.CoordDataset(testdata2, target_np=settings["target_np"])
    
  elif "gnina_resnet" in settings["model"]:
    model = fused.gnina_resnet(1, output_dim)
    data1 = feater.dataloader.VoxelDataset(datafile1)
    data2 = feater.dataloader.HilbertCurveDataset(datafile2)

    data1_test = feater.dataloader.VoxelDataset(testdata1)
    data2_test = feater.dataloader.HilbertCurveDataset(testdata2)

  elif "pointnet_resnet" in settings["model"]:
    model = fused.pointnet_resnet(1, output_dim)
    if settings["model"].endswith("_s"): 
      data1 = feater.dataloader.SurfDataset(datafile1, target_np=settings["target_np2"])
      data1_test = feater.dataloader.SurfDataset(testdata1, target_np=settings["target_np2"])
    else: 
      data1 = feater.dataloader.CoordDataset(datafile1, target_np=settings["target_np"])
      data1_test = feater.dataloader.CoordDataset(testdata1, target_np=settings["target_np"])

    data2 = feater.dataloader.HilbertCurveDataset(datafile2)
    data2_test = feater.dataloader.HilbertCurveDataset(testdata2)

  elif "pointnet_pointnet" in settings["model"]:
    model = fused.pointnet_pointnet(1, output_dim)
    data1 = feater.dataloader.CoordDataset(datafile1, target_np=settings["target_np"])
    data1_test = feater.dataloader.CoordDataset(testdata1, target_np=settings["target_np"])

    if settings["model"].endswith("_s"): 
      data2 = feater.dataloader.SurfDataset(datafile2, target_np=settings["target_np2"])
      data2_test = feater.dataloader.SurfDataset(testdata2, target_np=settings["target_np2"])
    else:
      print("Warning: The second pointnet dataset is REALLY another coordinate dataset??? ")
      data2 = feater.dataloader.CoordDataset(datafile2, target_np=settings["target_np2"])
      data2_test = feater.dataloader.CoordDataset(testdata2, target_np=settings["target_np2"])

  else:
    raise ValueError("Model not found")

  model = model.cuda()

  c = 0
  for m in model.modules():
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
      print(f"Init Conv Layer {c}")
      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      if m.bias is not None:
        nn.init.constant_(m.bias, 0)
      c += 1
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
      print(f"Init BatchNorm Layer {c}")
      nn.init.constant_(m.weight, 1)
      nn.init.constant_(m.bias, 0)
      c += 1
    elif isinstance(m, nn.Linear):
      print(f"Init Linear Layer {c}")
      nn.init.normal_(m.weight, 0, 0.1)
      if m.bias is not None:
        nn.init.constant_(m.bias, 0)
      c += 1

  if len(data1) != len(data2): 
    raise ValueError("Data1 and data2 have different length") 

  perform_training(settings, model, data1, data2, data1_test, data2_test)


"""
/Weiss/clustered_single/TestSet_NonRedund.h5
/Weiss/clustered_single/TestSet_NonRedund_hilb.h5
/Weiss/clustered_single/TestSet_NonRedund_surf.h5
/Weiss/clustered_single/TestSet_NonRedund_vox.h5
/Weiss/clustered_single/TrainingSet_NonRedund.h5
/Weiss/clustered_single/TrainingSet_NonRedund_hilb.h5
/Weiss/clustered_single/TrainingSet_NonRedund_surf.h5
/Weiss/clustered_single/TrainingSet_NonRedund_vox.h5
"""

# gnina_pointnet: 30 epochs ~75% accuracy required Learning rate being 0.0001. 
# python train_fused_models.py -m gnina_pointnet -train /Weiss/clustered_single/TrainingSet_NonRedund_vox.h5%/Weiss/clustered_single/TrainingSet_NonRedund.h5 -test /Weiss/clustered_single/TestSet_NonRedund_vox.h5%/Weiss/clustered_single/TestSet_NonRedund.h5 -o /tmp/ -w 12 --test-number 4000 -b 128 -e 1 -lr 0.0001 --lr-decay-steps 30 --lr-decay-rate 0.5 --target_np 24 


# gnina_resnet: 30 epochs ~ 95% accuracy
# python train_fused_models.py -m gnina_resnet -train /Weiss/clustered_single/TrainingSet_NonRedund_vox.h5%/Weiss/clustered_single/TrainingSet_NonRedund_hilb.h5 -test /Weiss/clustered_single/TestSet_NonRedund_vox.h5%/Weiss/clustered_single/TestSet_NonRedund_hilb.h5 -o /tmp/ -w 12 --test-number 4000 -b 128 -e 1 -lr 0.0001 --lr-decay-steps 30 --lr-decay-rate 0.5 


# pointnet_resnet: 30 epochs ~ 90% accuracy
# python train_fused_models.py -m pointnet_resnet -train /Weiss/clustered_single/TrainingSet_NonRedund.h5%/Weiss/clustered_single/TrainingSet_NonRedund_hilb.h5 -test /Weiss/clustered_single/TestSet_NonRedund.h5%/Weiss/clustered_single/TestSet_NonRedund_hilb.h5 -o /tmp/ -w 12 --test-number 4000 -b 128 -e 1 -lr 0.0001 --lr-decay-steps 30 --lr-decay-rate 0.5 --target_np 24


# pointnet_pointnet: 30 epochs ~ 50% accuracy 
# python train_fused_models.py -m pointnet_pointnet -train /Weiss/clustered_single/TrainingSet_NonRedund.h5%/Weiss/clustered_single/TrainingSet_NonRedund_surf.h5 -test /Weiss/clustered_single/TestSet_NonRedund.h5%/Weiss/clustered_single/TestSet_NonRedund_surf.h5 -o /tmp/ -w 12 --test-number 4000 -b 128 -e 1 -lr 0.0001 --lr-decay-steps 30 --lr-decay-rate 0.5 --target_np 24 --target_np2 1500